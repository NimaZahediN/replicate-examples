this is the huggingface repo where /workspaces/replicate-examples/mistral-transformers-2/predict.py references the model:

``` 
Model Description
A model that can generate Honeycomb Queries.
This model is a fine-tuned version of mistralai/Mistral-7B-v0.1.

fine-tuned by Hamel Husain

Usage
You can use this model with the following code:

First, download the model

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model_id='parlance-labs/hc-mistral-alpaca'
model = AutoPeftModelForCausalLM.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

Then, construct the prompt template like so:

def prompt(nlq, cols):
    return f"""Honeycomb is an observability platform that allows you to write queries to inspect trace data. You are an assistant that takes a natural language query (NLQ) and a list of valid columns and produce a Honeycomb query.

### Instruction:

NLQ: "{nlq}"

Columns: {cols}

### Response:
"""

def prompt_tok(nlq, cols):
    _p = prompt(nlq, cols)
    input_ids = tokenizer(_p, return_tensors="pt", truncation=True).input_ids.cuda()
    out_ids = model.generate(input_ids=input_ids, max_new_tokens=5000, 
                          do_sample=False)
    return tokenizer.batch_decode(out_ids.detach().cpu().numpy(), 
                                  skip_special_tokens=True)[0][len(_p):]

Finally, you can get predictions like this:

# model inputs
nlq = "Exception count by exception and caller"
cols = ['error', 'exception.message', 'exception.type', 'exception.stacktrace', 'SampleRate', 'name', 'db.user', 'type', 'duration_ms', 'db.name', 'service.name', 'http.method', 'db.system', 'status_code', 'db.operation', 'library.name', 'process.pid', 'net.transport', 'messaging.system', 'rpc.system', 'http.target', 'db.statement', 'library.version', 'status_message', 'parent_name', 'aws.region', 'process.command', 'rpc.method', 'span.kind', 'serializer.name', 'net.peer.name', 'rpc.service', 'http.scheme', 'process.runtime.name', 'serializer.format', 'serializer.renderer', 'net.peer.port', 'process.runtime.version', 'http.status_code', 'telemetry.sdk.language', 'trace.parent_id', 'process.runtime.description', 'span.num_events', 'messaging.destination', 'net.peer.ip', 'trace.trace_id', 'telemetry.instrumentation_library', 'trace.span_id', 'span.num_links', 'meta.signal_type', 'http.route']

# print prediction
out = prompt_tok(nlq, cols)
print(nlq, '\n', out)

This will give you a prediction that looks like this:

"{'breakdowns': ['exception.message', 'exception.type'], 'calculations': [{'op': 'COUNT'}], 'filters': [{'column': 'exception.message', 'op': 'exists'}, {'column': 'exception.type', 'op': 'exists'}], 'orders': [{'op': 'COUNT', 'order': 'descending'}], 'time_range': 7200}"

Alternatively, you can play with this model on Replicate: hamelsmu/honeycomb-2

Hosted Inference
This model is hosted on Replicate: (hamelsmu/honeycomb-2)[https://replicate.com/hamelsmu/honeycomb-2], using this config.

Training Procedure
Used axolotl, see this config. See this wandb run to see training metrics.

Framework versions
PEFT 0.7.0
Transformers 4.37.0.dev0
Pytorch 2.1.0
Datasets 2.15.0
Tokenizers 0.15.0

```

