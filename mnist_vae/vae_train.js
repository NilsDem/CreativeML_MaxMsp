/**
 * VAE training + inference helpers for browser-side MNIST.
 *
 * Architecture is aligned with the local notebook intent (2D latent, conv encoder/decoder)
 * but implemented in TensorFlow.js layers.
 */

import { IMAGE_H, IMAGE_W } from "./data.js";

export const LATENT_DIM = 2;
const SAVE_PREFIX = "mnist-vae-v1";

function assertTfLoaded() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js is not loaded. Include tf.min.js before loading modules.");
  }
}

function makeEncoder() {
  assertTfLoaded();

  const input = tf.input({ shape: [IMAGE_H, IMAGE_W, 1] });
  let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: "same" }).apply(input);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

  x = tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: "same" }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

  x = tf.layers.flatten().apply(x);
  x = tf.layers.dense({ units: 128, activation: "relu" }).apply(x);

  const zMean = tf.layers.dense({ units: LATENT_DIM, name: "z_mean" }).apply(x);
  const zLogVar = tf.layers.dense({ units: LATENT_DIM, name: "z_log_var" }).apply(x);

  return tf.model({ inputs: input, outputs: [zMean, zLogVar], name: "encoder" });
}

function makeDecoder() {
  assertTfLoaded();

  const z = tf.input({ shape: [LATENT_DIM] });

  let x = tf.layers.dense({ units: 7 * 7 * 64, activation: "relu" }).apply(z);
  x = tf.layers.reshape({ targetShape: [7, 7, 64] }).apply(x);

  x = tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: "same" }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

  x = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: "same" }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

  const logits = tf.layers.conv2dTranspose({
    filters: 1,
    kernelSize: 3,
    strides: 1,
    padding: "same",
    name: "decoder_logits",
  }).apply(x);

  return tf.model({ inputs: z, outputs: logits, name: "decoder" });
}

function sampleLatent(zMean, zLogVar) {
  return tf.tidy(() => {
    const eps = tf.randomNormal(zMean.shape);
    const std = zLogVar.mul(0.5).exp();
    return zMean.add(std.mul(eps));
  });
}

function reconstructionLoss(xTrue, logits) {
  return tf.tidy(() => {
    // Stable BCE from logits per pixel:
    // max(logit, 0) - logit * y + log(1 + exp(-abs(logit)))
    // Keeps shape [B, H, W, C] so sum over spatial dims is always valid.
    const perPixel = tf
      .maximum(logits, tf.scalar(0))
      .sub(logits.mul(xTrue))
      .add(tf.log(tf.add(tf.scalar(1), tf.exp(logits.abs().neg()))));
    const perExample = perPixel.sum([1, 2, 3]);
    return perExample.mean();
  });
}

function klLoss(zMean, zLogVar) {
  return tf.tidy(() => {
    const term = tf.onesLike(zLogVar).add(zLogVar).sub(zMean.square()).sub(zLogVar.exp());
    const perExample = term.sum(1).mul(-0.5);
    return perExample.mean();
  });
}

function getTrainableVars(encoder, decoder) {
  return [
    ...encoder.trainableWeights.map((w) => w.val),
    ...decoder.trainableWeights.map((w) => w.val),
  ];
}

export async function trainVAE({
  data,
  epochs = 10,
  stepsPerEpoch = 250,
  batchSize = 128,
  learningRate = 1e-3,
  klWeight = 1.0,
  onLog = () => {},
  onStepEnd = () => {},
  onEpochEnd = () => {},
} = {}) {
  assertTfLoaded();

  if (!data) {
    throw new Error("trainVAE requires a loaded MnistData instance.");
  }

  const encoder = makeEncoder();
  const decoder = makeDecoder();
  const optimizer = tf.train.adam(learningRate);
  const trainableVars = getTrainableVars(encoder, decoder);

  onLog(`Encoder params: ${encoder.countParams()}`);
  onLog(`Decoder params: ${decoder.countParams()}`);

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    let epochRecon = 0;
    let epochKl = 0;
    let epochTotal = 0;

    for (let step = 0; step < stepsPerEpoch; step += 1) {
      const batch = data.nextTrainBatch(batchSize);
      const xs = batch.xs.reshape([batchSize, IMAGE_H, IMAGE_W, 1]);

      let reconTensor;
      let klTensor;

      const { value, grads } = tf.variableGrads(() => {
        const [zMean, zLogVar] = encoder.apply(xs, { training: true });
        const z = sampleLatent(zMean, zLogVar);
        const logits = decoder.apply(z, { training: true });

        const recon = reconstructionLoss(xs, logits);
        const kl = klLoss(zMean, zLogVar);

        reconTensor = tf.keep(recon);
        klTensor = tf.keep(kl);

        return recon.add(kl.mul(klWeight));
      }, trainableVars);

      optimizer.applyGradients(grads);

      const [totalVal] = await value.data();
      const [reconVal] = await reconTensor.data();
      const [klVal] = await klTensor.data();

      epochTotal += totalVal;
      epochRecon += reconVal;
      epochKl += klVal;

      await onStepEnd({
        epoch,
        step,
        totalLoss: totalVal,
        reconLoss: reconVal,
        klLoss: klVal,
      });

      tf.dispose([value, reconTensor, klTensor, xs, batch.xs, batch.labels]);
      Object.values(grads).forEach((g) => g.dispose());

      if (step % 6 === 0) {
        await tf.nextFrame();
      }
    }

    const avgTotal = epochTotal / stepsPerEpoch;
    const avgRecon = epochRecon / stepsPerEpoch;
    const avgKl = epochKl / stepsPerEpoch;

    onLog(
      `Epoch ${epoch + 1}/${epochs} | total=${avgTotal.toFixed(3)} recon=${avgRecon.toFixed(3)} kl=${avgKl.toFixed(3)}`,
    );

    await onEpochEnd({
      epoch,
      totalLoss: avgTotal,
      reconLoss: avgRecon,
      klLoss: avgKl,
      encoder,
      decoder,
    });
  }

  return { encoder, decoder };
}

export async function saveVAE(encoder, decoder, prefix = SAVE_PREFIX) {
  assertTfLoaded();

  await encoder.save(`indexeddb://${prefix}-encoder`);
  await decoder.save(`indexeddb://${prefix}-decoder`);
}

export async function loadVAE(prefix = SAVE_PREFIX) {
  assertTfLoaded();

  const encoder = await tf.loadLayersModel(`indexeddb://${prefix}-encoder`);
  const decoder = await tf.loadLayersModel(`indexeddb://${prefix}-decoder`);
  return { encoder, decoder };
}

export function decodePoint(decoder, z0, z1) {
  assertTfLoaded();

  return tf.tidy(() => {
    const z = tf.tensor2d([[z0, z1]]);
    const logits = decoder.predict(z);
    return tf.sigmoid(logits).reshape([IMAGE_H, IMAGE_W]);
  });
}

export async function encodeLabeledBatch(encoder, xs2d, labelsOneHot) {
  assertTfLoaded();

  const batchSize = xs2d.shape[0];
  const xs4d = xs2d.reshape([batchSize, IMAGE_H, IMAGE_W, 1]);

  const outputs = encoder.predict(xs4d);
  const zMean = Array.isArray(outputs) ? outputs[0] : outputs;
  const zLogVar = Array.isArray(outputs) ? outputs[1] : null;
  const zArray = await zMean.array();
  const labelTensor = labelsOneHot.argMax(1);
  const labels = await labelTensor.array();

  tf.dispose([xs4d, zMean, zLogVar, labelTensor]);

  return zArray.map((pt, i) => [pt[0], pt[1], labels[i]]);
}

export async function encodeTestSetSample(encoder, data, count = 2000) {
  const batch = data.nextTestBatch(count);
  const points = await encodeLabeledBatch(encoder, batch.xs, batch.labels);
  tf.dispose([batch.xs, batch.labels]);
  return points;
}
