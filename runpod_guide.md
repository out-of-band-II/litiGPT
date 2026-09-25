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
is pinned to the physical host.** If that host's GPUs are all taken while your
pod is stopped, the pod will not start:

> There are not enough free GPUs on the host machine to start this pod.

That is not hypothetical — `v0u52o4sega7gs` hit it, and so did
`ams6n8th37ih4u` on 2026-09-22, the morning after a finished 3h run whose
adapter had not been pulled down yet.

**It is recoverable, and this guide used to say it was not.** A GPU pod can be
started **CPU-only** from the web console. That needs no free GPU on the host,
mounts the same `/workspace`, and bills about half — $0.37/hr against $0.74
for the 4090. SSH in, pull everything, stop it again. The rescue above cost
about two cents.

Two things to know before you rely on it:

- **The API cannot do it.** `pod-action` takes only
  `start`/`stop`/`restart`/`terminate`, and the update PATCH has no GPU-count
  field. The CPU-only start comes from the console or nowhere.
- **The address changes.** After it comes up, `get-pod` reports `gpu.count: 0`
  and a fresh `ssh.direct` host and port. Re-read them; do not trust the alias
  in `~/.ssh/config`, which will still hold the previous port.

A genuine host *failure*, as opposed to a busy host, is still unrecoverable.
So the rule stands, it is just no longer the only line of defence:

**Rule: pull anything you would mind losing before you stop a pod, not after.**
A LoRA adapter is 35–70MB. There is no excuse. Better still, put it on a
network volume, which survives the host outright.

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

> New to SSH? **[Appendix A](#appendix-a-ssh-from-first-principles)** explains
> keypairs, `known_hosts`, `~/.ssh/config`, tunnels and how to read the error
> messages. This section assumes it.

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

**One consequence: the `ssh.runpod.io` proxy stops working for you.** RunPod
offers two routes in, and they check different things:

| Route | Authenticates against | Carries scp |
|---|---|---|
| Direct, `root@<ip> -p <port>` | the pod's own `PUBLIC_KEY` | yes |
| Proxy, `<podid>-<hash>@ssh.runpod.io` | your **account-registered** keys | no |

A per-pod key is deliberately not registered on the account, so the proxy
refuses it — verified here, `Permission denied (publickey)`. That is the
feature working, not a fault. Use the direct route, which needs `22/tcp` in
the pod's exposed ports, and is the only route that carries file transfer
anyway.

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

**When the commits you need are not pushed yet**, a clone gets you the old
code, which is worse than useless when the whole point of the run is a fix you
just made. Send the history instead of publishing it:

```bash
git bundle create /tmp/litigpt.bundle main      # entire history, one file
scp /tmp/litigpt.bundle litigpt-pod:/workspace/
ssh litigpt-pod "cd /workspace && git clone litigpt-full.bundle litiGPT"
```

A bundle is a real git repository in a single file — the pod gets genuine
history and a working `git log`, nothing is fetched from the internet, and
nothing is published. The whole of this repo is about 400KB that way. To send
only new commits onto an existing clone, bundle a range (`git bundle create
f.bundle <base>..main`) and `git pull` the bundle on the pod.

**Data and models: they are gitignored, so they have to be copied.** For a
single file, `scp`. For a directory, tar over ssh is faster:

```bash
tar czf - models/litigpt_top30_lora data/training/val.jsonl \
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

In the **Create API key** dialog, choose **Restricted** and set:

| Scope | Value | Why |
|---|---|---|
| `api.runpod.io/graphql` | **Read / Write** | `runpodctl`'s pod commands go here |
| `api.runpod.ai` | **None** | the serverless API; the watchdog never touches it |

At least one permission must be granted or the dialog refuses to create the
key.

**Be clear about what this does and does not buy you.** There is no per-pod
scope, and no "Pods" scope: the granularity is the whole GraphQL API or
nothing. `Read / Write` there means write access to **every pod on the
account**, so a key sitting on this pod can stop or terminate a different
project's pod. Restricted is better than "All" — it withholds the serverless
API and the rest — but it does not isolate projects, and nothing on offer
does.

Two things follow:

- **Treat any pod holding a key as able to control the whole account.** That
  is an argument for not exposing such a pod publicly, and for deleting the
  key when the run is over.
- **RunPod already injects one.** Every pod's PID 1 environment carries a
  `RUNPOD_API_KEY`, readable by anyone with a shell on the pod. So a pod is
  arguably in this position already; a key you create yourself at least has a
  scope you chose and can revoke from the console.

Revoke it in the same **Settings → API Keys** page when the run is done.

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
# after a serving session, the artefact is the blind-eval log, not the adapter
scp litigpt-pod:/workspace/litiGPT/data/eval/blind_eval.jsonl ./data/eval/blind_eval.pod.jsonl

# after a training run, the adapter and the metrics DB
mkdir -p ./models/litigpt_top30_lora
tar czf - -C /workspace/litiGPT/models/litigpt_top30_lora . \
  | tar xzf - --no-same-owner -C ./models/litigpt_top30_lora
scp litigpt-pod:/workspace/mlflow.db ./models/litigpt_top30_lora/run_artifacts/
```

---

# Appendix A: SSH from first principles

Everything above leans on SSH. If you have been copying the commands without a
clear picture of what they do, this appendix is the picture. It is written for
someone who has never set up a key.

## What SSH is

SSH — Secure Shell — gives you a shell on a machine somewhere else. You type,
the remote machine runs it, you see the output. That is the whole idea, and it
predates the cloud entirely.

The *Secure* is the point. Its predecessor, telnet, sent everything as plain
text across the network, including your password. SSH encrypts the channel, so
anyone sitting between you and the pod sees only ciphertext.

But encryption alone is not enough. An encrypted channel to an imposter is
still a channel to an imposter. So SSH authenticates **both ends**:

- the **server** proves to you that it is the machine you meant to reach
- **you** prove to the server that you are allowed in

Those are two separate mechanisms, they fail with two different error
messages, and confusing them is the single biggest source of SSH frustration.

## Why keys instead of a password

A password is a shared secret: you know it, the server knows it, and you send
it over on every login. That means the server stores something that can be
stolen, and a machine that can impersonate the server can harvest it.

Key authentication removes the sharing. You generate a **keypair** — two files
that are mathematically linked:

| File | Name | Where it goes | Secrecy |
|---|---|---|---|
| `id_litigpt` | private key | stays on your laptop, always | never leaves, never copied |
| `id_litigpt.pub` | public key | copied onto every server you want to enter | harmless to publish |

The asymmetry is the trick. The public key can *verify* a signature but cannot
*produce* one. So the server can hold your public key, check that you possess
the matching private key, and still learn nothing that would let it — or
anyone who steals its disk — log in as you elsewhere.

**Your private key never crosses the network.** Not on setup, not on login,
not ever. If some tool asks you to paste a private key somewhere, that is
wrong.

## What actually happens when you connect

Roughly, in order:

1. **Key exchange.** The two sides agree on a shared session secret using
   Diffie-Hellman, without ever transmitting it. Everything after this point
   is encrypted.
2. **Server authentication.** The server signs a value with its *host key* and
   sends the signature. Your client checks it against `~/.ssh/known_hosts`.
   This is where "authenticity of host ... can't be established" comes from.
3. **Your authentication.** Your client offers a public key. The server looks
   for it in `~/.ssh/authorized_keys` on the account you are logging into. If
   it is there, the server sends a challenge; your client signs it with the
   private key; the server verifies with the public key.
4. **Shell.** You get a prompt.

On RunPod, step 3's `authorized_keys` is populated for you: whatever you put
in the pod's `PUBLIC_KEY` environment variable at creation is written there
when the container starts. That is why the variable only works at creation
time, and why retrofitting a key means recreating the pod.

## Host keys, `known_hosts`, and the scary warning

Host keys are the *server proving itself to you*, which is the direction
people forget exists.

The first time you connect, your client has never seen this machine and asks:

```
The authenticity of host '[38.65.239.38]:19226' can't be established.
ED25519 key fingerprint is SHA256:xxxx...
Are you sure you want to continue connecting (yes/no)?
```

Saying yes records the fingerprint in `~/.ssh/known_hosts`. On every later
connection the client compares silently. If it ever differs, you get:

```
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
```

In general that means something is impersonating the server. **On RunPod it
usually means something mundane**: pods are reached through shared IP and port
combinations, and the one you were given has been recycled to a different
machine. When you have just terminated a pod and created a new one, this is
expected. Remove the stale entry and reconnect:

```bash
ssh-keygen -R '[38.65.239.38]:19226'
```

Do that only when you *know* why the key changed. On a machine that should
have been stable, treat the warning as real.

## Where everything lives

All of it is in `~/.ssh/`, which on your Windows box is
`C:\Users\<you>\.ssh\`:

| File | What it is |
|---|---|
| `id_litigpt` | a private key — this project's |
| `id_litigpt.pub` | the matching public key |
| `known_hosts` | fingerprints of servers you have accepted |
| `config` | nicknames and per-host settings |
| `authorized_keys` | *on the server*: public keys allowed to log in |

Permissions matter. On Linux, SSH refuses to use a private key that other
users can read — it will error rather than silently proceed, which is correct
behaviour and surprises everyone once. Windows OpenSSH does the equivalent
check against NTFS ACLs; if you ever copy a key in and get a permissions
complaint, `icacls` is the fix, not `chmod`.

## `~/.ssh/config`: giving machines names

Without it, every command is a pile of flags:

```bash
ssh -i ~/.ssh/id_litigpt -p 19226 root@38.65.239.38
```

With it, the same connection is `ssh litigpt-pod`. This project's entry:

```
Host litigpt-pod
    HostName 38.65.239.38
    User root
    Port 19226
    IdentityFile ~/.ssh/id_litigpt
    IdentitiesOnly yes
```

Line by line:

- **`Host`** — the nickname you type. Purely local; it means nothing to the
  server.
- **`HostName`** — the real address it expands to.
- **`User`** — who to log in as. RunPod containers run as `root`.
- **`Port`** — SSH defaults to 22, but RunPod maps your pod's 22 to a
  high-numbered port on a shared host, so this is almost never 22.
- **`IdentityFile`** — which private key to use.
- **`IdentitiesOnly yes`** — *only* that key. Without it, your client offers
  every key it has, one at a time, and servers commonly cut you off after five
  failures — so you can be locked out while holding the correct key, simply
  because it was offered sixth. Always set this.

`HostName` and `Port` change every time a pod is recreated. `scp`, `rsync`,
`git` over SSH and `-L` tunnels all read this file, so one edit fixes
everything at once.

Two entries can point at the same machine — this project uses `litigpt-pod`
and `litigpt-eval` — which is convenient and is also exactly how the alias
accident described in section 4 happens. Keep a project's name in its alias.

## The agent

`ssh-agent` is a background process that holds unlocked private keys, so a
passphrase-protected key does not prompt you on every single connection. On
Windows it is a service you can start once:

```powershell
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_litigpt
```

The keys in this project have no passphrase, so you do not need the agent at
all. It matters the moment you add one — which you should, for any key that
guards something you care about.

## Moving files

Same authentication, different front end:

```bash
scp file.txt litigpt-pod:/workspace/          # push one file
scp litigpt-pod:/workspace/out.json .         # pull one file
scp -r localdir litigpt-pod:/workspace/       # push a directory
```

`scp` is fine for one or two files. It has two limits worth knowing:

- It does **not** expand shell braces remotely — `{a,b,c}` arrives as a
  literal filename and fails. That bit this project.
- It restarts from zero if interrupted.

For many files, pipe `tar` through SSH instead. SSH is just a pipe, so
anything that reads and writes streams works over it:

```bash
ssh litigpt-pod 'tar czf - -C /workspace/litiGPT/models final' \
  | tar xzf - --no-same-owner -C ./models/litigpt_top30_lora
```

`--no-same-owner` matters: the archive records root's uid, your Windows user
is not root, and without the flag every file errors on ownership.

For anything large or resumable, `rsync -avP` over the same alias beats both.

## Port forwarding, the part that looks like magic

This is how you reached the Gradio UI, and it is worth understanding properly
rather than pattern-matching.

```bash
ssh -N -L 7870:localhost:7870 litigpt-pod
```

Read `-L A:B:C` as: **open port A on my machine; anything that connects to it
comes out of the SSH connection and is delivered to B:C, resolved from the
remote machine's point of view.**

So, step by step:

1. Your laptop starts listening on port 7870.
2. Your browser connects to `http://localhost:7870`.
3. Those bytes travel the encrypted SSH connection to the pod's SSH daemon.
4. The daemon opens a fresh connection to `localhost:7870` — **`localhost`
   here means the pod**, not you. That is the part that trips people up.
5. Gradio answers, and the bytes come back the same way.

Why that is the right tool here: the app binds to `127.0.0.1` on the pod, so
the pod's own firewall and the internet cannot reach it at all. The only door
is the tunnel, and the tunnel is guarded by your SSH key. No password on the
app, no public URL, nothing to leak.

The two flags:

- **`-N`** — do not run a remote command. Without it you also get an
  interactive shell, and closing it kills the tunnel. `-N` says the forwarding
  *is* the job.
- **`-o ExitOnForwardFailure=yes`** — if local port 7870 is already taken, SSH
  would otherwise connect happily and just not forward anything, leaving you
  staring at a browser error with a session that looks healthy. This makes it
  fail loudly.

Add `-o ServerAliveInterval=30` for long-lived tunnels; it sends a keepalive
so an idle NAT or firewall does not quietly drop the connection.

The mirror image is `-R` (remote forwarding), which opens a port on the
*server* that reaches back to your machine. You have not needed it here.

## Reading failures

Add `-v` for a running commentary, `-vvv` for more than you want:

```bash
ssh -v litigpt-pod
```

The useful lines are `debug1: Offering public key:` (which key was tried) and
`debug1: Authentications that can continue:` (what the server will accept).

| Message | What it means | Where to look |
|---|---|---|
| `Connection timed out` | Nothing answered. Pod stopped, or wrong IP/port. | Is it running? Re-read the address from the console. |
| `Connection refused` | The machine answered but nothing is listening on that port. | Usually the wrong port, or sshd not started. |
| `Permission denied (publickey)` | You reached the right sshd; your key is not in its `authorized_keys`. | Wrong `IdentityFile`, or the pod was made without your `PUBLIC_KEY`. |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Host key differs from `known_hosts`. | Expected after recreating a pod; `ssh-keygen -R`. |
| `Too many authentication failures` | Agent offered too many keys before the right one. | Set `IdentitiesOnly yes`. |
| `Bad owner or permissions` | Key file is too readable. | `chmod 600`, or `icacls` on Windows. |

The distinction that saves the most time: **`Permission denied` means you got
all the way to the right server.** The network is fine, the address is right,
the pod is up. It is purely about which key was offered. `timed out` and
`refused` are the opposite — you never got there at all.

## Flags used in this guide

| Flag | Meaning |
|---|---|
| `-i <file>` | Use this private key. |
| `-p <port>` | Connect to this port (note: `scp` uses capital `-P`). |
| `-N` | No remote command; forwarding only. |
| `-L a:h:p` | Local port forward, described above. |
| `-o BatchMode=yes` | Never prompt for anything; fail instead. Correct for scripts and automation, which is why it appears throughout this project. |
| `-o ExitOnForwardFailure=yes` | Fail if a tunnel cannot be established. |
| `-o ServerAliveInterval=30` | Keepalive every 30s for long-lived connections. |
| `-v` / `-vvv` | Increasing verbosity when something is wrong. |

## The three habits worth keeping

1. **One key per project, never the account-wide one.** A misdirected alias
   then fails with `Permission denied` instead of succeeding on the wrong
   machine. Section 4 covers the setup.
2. **Verify identity after connecting**, before running anything that writes:
   `ssh litigpt-pod "hostname; ls -d /workspace/litiGPT"`.
3. **Prefer a tunnel to a public port.** It costs one extra terminal and
   removes the entire question of who else can reach your app.

---

# Appendix B: Using a pod from VS Code or MobaXterm

Both work, and both are just SSH underneath. Everything in Appendix A still
applies: the same key, the same `~/.ssh/config` entry, the same host-key
warning when a pod is recreated. Neither tool needs its own credentials.

## One thing to settle first: direct SSH, not the proxy

RunPod offers two ways in, and they authenticate differently:

| Route | Address | Which key it accepts | Carries scp/sftp |
|---|---|---|---|
| **Direct** | `root@<ip> -p <high port>` | the pod's own `PUBLIC_KEY` | yes |
| **Proxy** | `<podid>-<hash>@ssh.runpod.io` | your **account-registered** keys | no |

This project sets a per-pod key (section 4) that is deliberately *not*
registered on the account, so the proxy refuses it:

```
ams6n8th37ih4u-6441108a@ssh.runpod.io: Permission denied (publickey).
```

That is correct, not a fault. **Use the direct route for everything below.**
It needs `22/tcp` in the pod's exposed ports, which this project always sets.
VS Code additionally cannot use the proxy at all, because it copies and runs a
server over the connection.

## VS Code Remote-SSH

This is the best of the options: you get the editor, a real terminal, and file
browsing against the pod filesystem, with local extensions where they belong.

**Install** the `Remote - SSH` extension (`ms-vscode-remote.remote-ssh`).

**Connect.** Because `~/.ssh/config` already has the alias, there is nothing to
configure:

1. `F1` -> **Remote-SSH: Connect to Host**
2. pick `litigpt-pod`
3. choose **Linux** if it asks what the platform is
4. `F1` -> **File: Open Folder** -> `/workspace/litiGPT`

The window reloads and the bottom-left corner turns green with the host name.
A terminal opened now (`` Ctrl+` ``) is a shell **on the pod**, in the right
directory, which removes a whole category of mistake -- there is no longer a
local terminal and a remote terminal that look identical.

**Point it at the venv.** `F1` -> **Python: Select Interpreter** ->
**Enter interpreter path** -> `/workspace/venv/bin/python`. Without this, the
editor resolves imports against the system interpreter, which has none of the
project dependencies, and every import is underlined in red while the code
runs perfectly.

**Forwarded ports come for free.** Start the Gradio app on the pod and VS Code
notices the listening socket and forwards it automatically; the **Ports** panel
next to the terminal lists it and turns it into a clickable
`http://localhost:...`. This is the same SSH local forward as Appendix A, set
up for you. You can also add one by hand in that panel, which is useful when
the app started before the window connected.

Two things to know:

- **The first connect is slow.** VS Code downloads and installs its server into
  `~/.vscode-server` on the pod, which is a minute or two. It lands on the
  container filesystem, so a pod **restart** wipes it and the next connect pays
  the cost again.
- **It keeps a connection open.** That is not a problem for the pod, but the
  pod is still billing whether or not you are typing. The editor being open is
  not what costs money; the pod being `RUNNING` is.

**When a pod is recreated**, update `HostName` and `Port` in `~/.ssh/config`
and reconnect. If VS Code hangs on connect after that, the usual cause is the
host-key change from Appendix A -- open a plain `ssh litigpt-pod` in a terminal
first, where the error is actually readable, and fix it there.

## MobaXterm

Windows-native, and its strength is the built-in SFTP browser: connect a
terminal and you get a file pane alongside it, drag-and-drop in both
directions, with no scp syntax to remember.

**Session** -> **SSH**, then:

| Field | Value |
|---|---|
| Remote host | the pod IP, e.g. `213.173.107.97` |
| Specify username | `root` |
| Port | the pod's mapped SSH port, e.g. `11216` |
| Advanced SSH settings -> Use private key | `C:\Users\<you>\.ssh\id_litigpt` |

MobaXterm reads OpenSSH keys directly; there is no need to convert anything to
PuTTY's `.ppk` format, and no need to run its key generator.

It can also read `~/.ssh/config` if you point it there
(**Settings -> Configuration -> SSH -> Use SSH config file**), after which the
alias works as it does everywhere else and you stop maintaining the address in
two places. Worth doing, given how often a pod address changes.

**Tunnels** are under **Tunneling** -> **New SSH tunnel** -> *Local port
forwarding*: listen on `7870` locally, forward to `localhost:7870` on the pod.
The same `-L` from Appendix A, drawn as a diagram, and the diagram is a genuine
help for remembering which side each hostname is resolved on.

## Which to use

- **VS Code** for anything involving editing code or reading logs while you
  work. The terminal being unambiguously on the pod is worth a lot on its own.
- **MobaXterm** for moving files around, or when you want a plain terminal
  without an editor attached.
- **Plain `ssh` from a terminal** for anything scripted, and for debugging a
  connection -- both GUIs hide the error text that tells you what is wrong.

None of them changes what is running on the pod. A training run started with
`setsid nohup` (section 7) keeps going when you close any of them, and a run
started in a VS Code terminal **without** `setsid nohup` dies when that window
disconnects, exactly as it would over plain SSH.
