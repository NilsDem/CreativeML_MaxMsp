/**
 * Adapted from the TensorFlow.js MNIST data loader pattern.
 * Added explicit nextTrainBatch/nextTestBatch APIs for browser-side VAE training.
 */

export const IMAGE_H = 28;
export const IMAGE_W = 28;
export const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

function assertTfLoaded() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js is not loaded. Include tf.min.js before loading modules.");
  }
}

export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.trainIndices = null;
    this.testIndices = null;
  }

  async load() {
    assertTfLoaded();

    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const imgRequest = new Promise((resolve) => {
      img.crossOrigin = "anonymous";
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i += 1) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize,
          );

          ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j += 1) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }

        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);

    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  getTrainData() {
    assertTfLoaded();
    const xs = tf.tensor4d(this.trainImages, [NUM_TRAIN_ELEMENTS, IMAGE_H, IMAGE_W, 1]);
    const labels = tf.tensor2d(this.trainLabels, [NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
    return { xs, labels };
  }

  getTestData(numExamples = null) {
    assertTfLoaded();
    let xs = tf.tensor4d(this.testImages, [NUM_TEST_ELEMENTS, IMAGE_H, IMAGE_W, 1]);
    let labels = tf.tensor2d(this.testLabels, [NUM_TEST_ELEMENTS, NUM_CLASSES]);

    if (numExamples != null) {
      xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
      labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
    }

    return { xs, labels };
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(batchSize, [this.trainImages, this.trainLabels], () => {
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
      return this.trainIndices[this.shuffledTrainIndex];
    });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, indexFn) {
    assertTfLoaded();
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i += 1) {
      const idx = indexFn();
      const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}
