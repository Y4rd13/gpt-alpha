# TODO

- [ ] Check the outputs shapes of the encoder layers. For example, in Add & Norm, the output shape is the same as the input shape. But in the Feed Forward layer, the output shape is different from the input shape. So, we need to check the output shape of each layer and make sure that the input shape of the next layer is the same as the output shape of the previous layer.

Shapes to check:

- [x] Positional Embedding
- [x] Attention
  - [x] filtered_value
- [x] Multi-Head Attention (MHA)
- [x] Dropout
- [x] Add & Norm
  - [x] Add
  - [x] Layer Normalization
- [x] Feed Forward
- [x] Dropout
- [x] Add & Norm
  - [x] Add
  - [x] Layer Normalization
