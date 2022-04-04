//import * from '@tensorflow/tfjs-node';
const tf = require('@tensorflow/tfjs-node');


// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
});


// console.log('here')
///**
// * Get the car data reduced to just the variables we are interested
// * and cleaned of missing data.
// */
//async function getData() {
//  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
//  const carsData = await carsDataResponse.json();
//  const cleaned = carsData.map(car => ({
//    mpg: car.Miles_per_Gallon,
//    horsepower: car.Horsepower,
//  }))
//  .filter(car => (car.mpg != null && car.horsepower != null));
//
//  return cleaned;
//}
//async function run() {
//  // Load and plot the original input data that we are going to train on.
//  const data = await getData();
//  const values = data.map(d => ({
//    x: d.horsepower,
//    y: d.mpg,
//  }));
//
//  tfvis.render.scatterplot(
//    {name: 'Horsepower v MPG'},
//    {values},
//    {
//      xLabel: 'Horsepower',
//      yLabel: 'MPG',
//      height: 300
//    }
//  );
//
//  // More code will be added below
//}
//
////document.addEventListener('DOMContentLoaded', run);
//run();
