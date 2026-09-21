# Working with RunPod on litiGPT

A practical guide to renting, driving and debugging GPU pods for this project.
Every example here is something that actually happened while training and
serving the top-30 model, including the mistakes.

---

## 1. The mental model

A pod is a container on someone's GPU machine. Three things have different
lifetimes, and most RunPod pain comes from confusing them:

| Lives in | Survives restart? | Survives host loss? | Use for |
|---|---|---|---|
| Container disk (`/`, `/root`) | **No** — wiped | No | Nothing you care about |
| Pod volume (`/workspace`) | Yes | **No** — pinned to the host | Model cache, venv, working data |
| Network volume | Yes | Yes | Anything you cannot re-create |
| Your own machine | — | — | The actual deliverables |

The row that bites is the third column of the pod volume. **Pod-local storage
is pinned to the physical host.** If that host's GPUs get taken while your pod
is stopped, you cannot restart it *and you cannot get your data off it*. The
volume is stranded with the machine.

That is not hypothetical — it happened to `v0u52o4sega7gs` here:

> Your Pod's GPUs are no longer available.

The only reason it cost nothing was that the adapter, checkpoints and MLflow
database had been copied down **before** the pod was stopped.

**Rule: pull anything you would mind losing before you stop a pod, not after.**
A LoRA adapter is 35–70MB. There is no excuse.

---

## 2. SECURE vs COMMUNITY — the CPU is the catch

This is the single least obvious thing about RunPod, and it cost an hour here.

- **SECURE** — RunPod's own datacenter hardware. Costs more.
- **COMMUNITY** — hardware hosted by third parties. Often half the price.

The GPU you rent is yours in both. **The host CPU is not.** On a busy
COMMUNITY host the CPU is heavily oversubscribed, and that wrecks
*token-by-token generation* even though the GPU is idle-fast.

Measured on the COMMUNITY 4090 (`h703y74fzh4xyk`):

```
bf16 matmul:     144 TFLOPS      <- GPU is genuinely excellent
mem bandwidth:   751 GB/s        <- normal
load average:    292  on 256 cores
python loop 3M:  4.17s           <- 20x slow, and no GPU involved at all
tiny-op launch:  99 us           <- 10x slow (healthy: 5-15 us)

result: 2.3 tokens/sec from a 3.8B model that should do 50+
```

Why the contradiction? A big matmul is *one* kernel launch and a lot of
compute — the GPU does all the work and the CPU is irrelevant. Decoding one
token at a time is *thousands* of tiny kernel launches, each dispatched by the
CPU. When the CPU is starved, generation becomes launch-bound and the GPU sits
at 0% waiting for work.

**Training is far more tolerant than inference.** Training runs big batched
kernels; a saturated host CPU barely shows. Serving a chat UI is the opposite.

**Rule: train wherever is cheap; serve interactively on SECURE.** A weaker
SECURE GPU will beat a starved COMMUNITY 4090 for chat.

---

## 3. Creating a pod

```
create-pod
  name            litigpt-eval
  image           runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
  gpu.id          NVIDIA GeForce RTX 4090
  gpu.allowedCudaVersions  ["12.8"]
  cloud           SECURE
  disk            20                          # container, ephemeral
  mounts.persistent  { size: 30, path: /workspace }
  ports           ["22/tcp", "7861/http"]
  startSsh        true
  env.PUBLIC_KEY  <your litigpt-only public key>
```

Decisions worth making deliberately:

**Pin the CUDA version.** `allowedCudaVersions: ["12.8"]` keeps you on hosts
matching your wheels. Availability is per-CUDA-version: a GPU can read
"available" overall and be out of stock on the version you need. Check with
`get-capacity` and a `cudaVersions` probe before assuming.

**Pick the image to match.** The old `runpod/pytorch:2.4.0-...-cuda12.4.1`
image needed hours of surgery — its torch was too old for transformers, and
upgrading torch broke its bundled torchvision. The
`1.0.2-cu1281-torch280` image ships torch 2.8.0+cu128 working out of the box.
**Choosing the right image is worth more than any amount of debugging.**

**Set `PUBLIC_KEY` at creation.** See §4 — this only works at creation.

**Ports must be declared here** (though they can be edited later). See §7 for
why the number matters more than you would expect.

---

## 4. SSH, and why per-pod keys matter

By default `startSsh: true` injects **your account's registered keys** into
every pod. If you run more than one project on one RunPod account, every pod
accepts the same key, and an alias pointed at the wrong pod does not fail —
**it connects**, and whatever you were about to run runs there. A file copy
lands on the wrong machine silently.

Setting `PUBLIC_KEY` explicitly at creation overrides that, giving the pod its
own key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_litigpt -N "" -C "litigpt-only"
# pass the .pub contents as env.PUBLIC_KEY when creating the pod
```

Verify it rather than trusting it:

```bash
# the account-wide key must be REFUSED
ssh -i ~/.ssh/id_runpod -o IdentitiesOnly=yes root@<ip> -p <port> true
#   -> Permission denied (publickey,password)   correct

# the project key must work
ssh -i ~/.ssh/id_litigpt -o IdentitiesOnly=yes root@<ip> -p <port> true
```

Now a misdirected alias is a permission error instead of a silent success.
**This cannot be retrofitted** — changing `PUBLIC_KEY` on a running pod does
not re-run the provisioning that installs it. It means recreating the pod.

### Aliases

Give every pod a project-specific alias. Never a generic `runpod`:

```
Host litigpt-pod
    HostName 38.65.239.38
    Port 19226
    User root
    IdentityFile ~/.ssh/id_litigpt
    IdentitiesOnly yes
    UserKnownHostsFile ~/.ssh/known_hosts_runpod
```

`UserKnownHostsFile` matters: pod host keys are new every time and RunPod
recycles IP:port pairs, so a shared `known_hosts` will eventually throw a
host-key-changed warning that looks like an attack but is just churn. Keeping
pod keys in their own file stops that noise from touching your real one.

**A stopped pod has no address.** `ssh.direct` is `null` until it starts, and
the IP and port usually **change** on every start. Re-read them from `get-pod`
and update the alias each time. An alias left pointing at a stale address is
how you end up connected to a different machine entirely.

Always confirm where you landed before doing anything destructive:

```bash
ssh litigpt-pod "hostname; ls -d /workspace/litiGPT"
```

---

## 5. Getting code and data onto a pod

**Code: clone it.** The repo is public, so no credentials are needed:

```bash
ssh litigpt-pod "cd /workspace && git clone https://github.com/<you>/litiGPT.git"
```

Use the **HTTPS** URL. The `origin` remote here is `git@github-anon:...`, where
`github-anon` is an alias defined in your *local* `~/.ssh/config` — a pod has
no idea what that means. (I skipped straight to `tar` on that basis without
checking whether HTTPS would work. It would have. Cloning is better: `git pull`
then updates the pod in one command.)

**Data and models: they are gitignored, so they have to be copied.** For a
single file, `scp`. For a directory, tar over ssh is faster:

```bash
tar czf - models/litigpt_top30_lora/final data/training/val.jsonl \
  | ssh litigpt-pod "tar xzf - --no-same-owner -C /workspace/litiGPT"
```

`--no-same-owner` is not optional. Without it every entry errors with
`Cannot change ownership to uid ...` because your local UID does not exist on
the pod. The files still extract, so it looks like noise, but the exit status
is failure and it will break any `set -e` script.

Watch out when the target directory already exists: `mv src dst` moves `src`
*inside* `dst` rather than replacing it. Cloning the repo creates `data/` from
its tracked `.gitkeep` files, so a later `mv _data data` silently produced
`data/_data/...` here.

---

## 6. Installing dependencies

Newer RunPod images are Ubuntu 24.04, where **PEP 668 blocks pip from
installing into the system interpreter**:

```
error: externally-managed-environment
```

Do not reach for `--break-system-packages`. Use a venv that inherits the
image's CUDA torch, so you do not re-download 2.5GB of it:

```bash
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install transformers==4.57.6 peft==0.21.0 ...
```

Put it in `/workspace` so it survives restarts. Then remember that plain
`python` is still the *system* interpreter with none of your packages —
scripts must call `/workspace/venv/bin/python` explicitly.

**Pin peft to the version that saved your adapter.** Loading a newer adapter
with an older peft prints `Unexpected keyword arguments [...] for class
LoraConfig` and quietly drops config it does not understand.

**Set `HF_HOME` onto the volume**, or the 7.6GB base model downloads again
after every restart:

```bash
export HF_HOME=/workspace/.cache/huggingface
```

---

## 7. Running things that outlive your SSH session

```bash
setsid nohup python -u -m litigpt.pipeline --step train \
  > /workspace/train.log 2>&1 < /dev/null &
echo $! > /workspace/train.pid
```

- **`setsid nohup`** — detaches from the session, so closing SSH does not kill it.
- **`python -u`** — Python block-buffers stdout when redirected. Without `-u`
  the log stays empty for hours and looks like a hang while tqdm (stderr,
  unbuffered) is the only thing coming through.
- **Write a PID file.** Then stop things by PID.

**Never stop things with `pkill -f <pattern>`.** The pattern matches the
command line of the shell you are typing it into, so it kills your own session
(exit 255). It also matches the `bash -c` wrapper that launched the real
process — a watchdog told to watch `litigpt.pipeline` latched onto the wrapper
instead of python, and would have fired the moment the wrapper exited rather
than when training ended.

Select processes precisely instead:

```bash
ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /litigpt/ {print $1}'
```

`scripts/pod_serve.sh` and `scripts/post_training_watchdog.sh` both follow
these rules.

---

## 8. Reaching a web UI on the pod

### The nginx N+1 convention

RunPod images run an nginx that listens on each exposed HTTP port and proxies
it to **the port below**:

```
/etc/nginx/nginx.conf:  listen 7861;  proxy_pass http://localhost:7860;
                        listen 8081;  proxy_pass http://localhost:8080;
                        listen 3001;  proxy_pass http://localhost:3000;
```

Two consequences:

1. **Your app goes on 7860, not 7861.** Binding 7861 fails with `EADDRINUSE`
   because nginx already holds it.
2. **An app on 7860 is automatically public** at
   `https://<POD_ID>-7861.proxy.runpod.net`, with no authentication.

This also produces a nasty false positive: `curl localhost:7861` returns
**200 from nginx** while your app is still loading. A readiness check that only
curls the port will declare success and then watch the app die on bind. Check
that *your* PID owns the socket:

```bash
ss -ltnp | grep "pid=$APP_PID,"
```

### Tunnel (private) vs proxy (public)

**SSH tunnel — the default, and what you want for yourself:**

```bash
ssh -N -L 7870:localhost:7870 litigpt-pod
# then open http://localhost:7870
```

Bind the app to `127.0.0.1` on a port nginx does not touch (7870 here).
Nothing is exposed to the internet.

**RunPod HTTP proxy — public, for other people:**

```
https://<POD_ID>-<PORT>.proxy.runpod.net
```

Bind to `0.0.0.0` on a port that is exposed and nginx-proxied. Then understand
what you have made: **the URL is public to anyone who has it.** The pod ID is
obscurity, not access control — RunPod's own documentation says so. For this
project the app impersonates real, named people, so always pair it with
`--auth user:pass`.

Also note the proxy **drops requests after 100 seconds**. Fine on a healthy
GPU, not fine on CPU or a starved host.

---

## 9. Money

- Billing runs from creation to stop/terminate, **whether or not anything is
  running**. A finished training job leaves the GPU idle at full price.
- **Nothing stops itself.** An idle 4090 is $0.34–0.74/hr, i.e. $8–18/day.
- **stop** releases the GPU and keeps the volume (storage still bills).
  **terminate** deletes everything.
- A stopped pod can only restart if that host has a free GPU. See §1.

Automate it: `scripts/post_training_watchdog.sh` waits for the training PID,
archives the adapter and calls `runpodctl stop pod`. It needs an API key on the
pod:

```bash
runpodctl config --apiKey <key>
```

Use a **Restricted** key with write access to Pods, never an "All" key — a key
sitting on a pod can terminate every other pod on the account, including other
projects'.

---

## 10. A debugging ladder

When something is slow or broken, work outward from the narrowest question.
The order matters: it stops you fixing code when the problem is the host.

**Is it my code, or the machine?** Benchmark the hardware directly:

```python
# GPU compute — a healthy 4090 gives 50-140 TFLOPS
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
# time 50 iterations of a @ a

# CPU — pure python, no GPU at all
t = time.time(); s = 0
for i in range(3_000_000): s += i     # healthy: 0.15-0.30s

# kernel launch latency
x = torch.randn(64, 64, device='cuda')
# time 2000 iterations of x + 1       # healthy: 5-15 us/op
```

Plus `uptime` — compare load average against `nproc`. Load 292 on 256 cores
means you are sharing with someone hungry.

**Fast GPU + slow CPU + slow launches = a starved host, not your bug.** Move
to SECURE. No amount of code tuning fixes it.

**Is the model actually where I think it is?**

```python
print(model.hf_device_map)
print({str(p.device) for p in model.parameters()})
```

`device_map="auto"` silently offloads whatever does not fit to **disk**, leaving
those parameters on the `meta` device. LoRA weights written to meta tensors are
discarded as a no-op — you get a model that loads, runs, and answers out of
the wrong weights. Watch for `Some parameters are on the meta device` and
`copying from a non-meta parameter ... which is a no-op`.

**Narrow generation slowness by layer:**

```
raw forward, 1 token   ->  is the model itself slow?
base model, no adapter ->  is it PEFT?
after merge_and_unload ->  is it the LoRA indirection?
generate() vs forward  ->  is it generate's overhead?
```

Here that sequence showed 347ms for the *base* model alone, which ruled out
the adapter entirely and pointed at the host.

**Did the adapter attach to what I meant?** Read the weights, not the config:

```python
from safetensors import safe_open
with safe_open("adapter_model.safetensors", framework="pt") as f:
    keys = list(f.keys())
```

`adapter_config.json` records what you *asked for*. The weights show what you
*got*. See §11.

**Does the error come from the model repo's own code?** A traceback through
`transformers_modules.<org>.<model>...` is running `modeling_*.py` downloaded
from the model repo, frozen at whatever transformers API existed when it was
uploaded. Phi-3's calls `DynamicCache.seen_tokens`, removed in newer
transformers:

```
AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
```

Fix by dropping `trust_remote_code=True` so transformers uses its own
maintained implementation.

---

## 11. Check that LoRA hit the modules you meant

`config.top30.yaml` asked for seven target modules:

```yaml
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

The trained adapter contains **two**:

```
mlp.down_proj       64 tensors
self_attn.o_proj    64 tensors
```

Phi-3 **fuses** its projections — `qkv_proj` instead of separate q/k/v, and
`gate_up_proj` instead of gate/up. Five of the seven names simply do not exist
in the architecture, and PEFT adapted the two that did without complaining.
Training ran fine and eval loss fell, so nothing ever surfaced it.

Those names are Llama's. They are not portable across architectures:

| Architecture | Attention | MLP |
|---|---|---|
| Llama / Mistral | `q_proj` `k_proj` `v_proj` `o_proj` | `gate_proj` `up_proj` `down_proj` |
| Phi-3 | `qkv_proj` `o_proj` | `gate_up_proj` `down_proj` |

Before a long run, print what the model actually has:

```python
print({n.split('.')[-1] for n, _ in model.named_modules() if n.endswith('_proj')})
```

---

## Quick reference

```bash
# connect (always project-specific alias, always verify)
ssh litigpt-pod "hostname; ls -d /workspace/litiGPT"

# update code
ssh litigpt-pod "cd /workspace/litiGPT && git pull"

# serve the blind eval, private via tunnel
ssh litigpt-pod "bash /workspace/litiGPT/scripts/pod_serve.sh blind"
ssh -N -L 7870:localhost:7870 litigpt-pod        # from your machine

# serve it publicly, with a login
ssh litigpt-pod "LITIGPT_PORT=7860 bash /workspace/litiGPT/scripts/pod_serve.sh \
    blind --host 0.0.0.0 --auth user:pass"

# stop the app / check on it
ssh litigpt-pod "bash /workspace/litiGPT/scripts/pod_serve.sh stop"
ssh litigpt-pod "tail -f /workspace/serve/app.log"

# pull results BEFORE stopping the pod
tar czf - -C /workspace/litiGPT/models/litigpt_top30_lora/final . \
  | tar xzf - --no-same-owner -C ./models/litigpt_top30_lora/final
```
