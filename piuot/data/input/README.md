Put the user-provided input file here, for example:

- `input.h5ad`

Expected content:
- one latent representation in `obsm`, such as `X_gae15` or `X_gaga15`
- if `data.embedding_key` is empty, the repo will look for `X_{method}{epoch}` from `reduction.method` and `reduction.epoch`
- one model-time field in `obs`, such as `time_bin`
- one raw-time field in `obs`, such as `t`

Optional downstream labels:
- `obs[state_key]`
- `obs[fate_key]`
