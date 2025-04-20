# `litiGPT`

- [`litiGPT`](#litigpt)
  - [Workflow:](#workflow)
  - [Data source](#data-source)


Folder with scripts to train a chatbot based on user interaction on the `r/litigi` subreddit

## Workflow:

1) Data
   1. Obtain and unpack data
   2. Assess data quality and do basic cleanup. Remove formatting artefacts, links, urls etc.
   3. Prepare comment chains. Format them to be fed into the tokenizer.


## Data source

Data taken from [academictorrents website](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13)

Contains reddit comments/submissions from 2005-06 to 2024-12.