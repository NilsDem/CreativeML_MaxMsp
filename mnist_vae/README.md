# MNIST VAE (Browser Training + Latent Explorer)

This app trains a 2D MNIST VAE directly in the browser with TensorFlow.js, then lets you explore the latent space interactively by hovering over the plot.

## Run

1. Open `mnist_vae/` in VS Code.
2. Start **Live Server** on `mnist_vae/index.html`.
3. Click:
   - `Load MNIST Data`
   - `Train VAE`
4. Hover over the latent plot to decode digits.

You can also save/load model weights with IndexedDB.

## Notes On Reused Local Code

The app intentionally reuses ideas from your local folder `mnist_vae/VAE-Latent-Space-Explorer-master` by copying/adapting behavior directly (no runtime link to that folder):

- Original public project:
  - https://github.com/tayden/VAE-Latent-Space-Explorer
- Hover-to-decode latent interaction pattern adapted from:
  - `src/components/XYPlot.jsx`
- Tensor-to-canvas digit rendering pattern adapted from:
  - `src/components/ImageCanvas.jsx`

Inline comments in `app.js` mark where these adaptations were applied.

## Architecture Reference

The VAE architecture in `vae_train.js` is aligned with the intent of your local notebook:

- `mnist_vae/VAE-Latent-Space-Explorer-master/scripts/VAE.ipynb`

(2D latent space, conv encoder/decoder, reconstruction + KL loss)
