// ---------------------------------------------------- BOF K-Means Clustering Algorithm ----------------------------------------------------------
/* var ejs_kmeans = require('./ejs_kmeans');
function TestData(samples,centroid){
	console.log('Samples Data : %s \n','('+samples.join(') (')+')');
	var k_means = new ejs_kmeans.k_mean_cluster(samples);
	k_means.initialize(centroid);
	k_means.calculate();
	console.log(k_means.result().replace(/<br \/>/g,"\n").replace(/&nbsp;/g,' ').replace(/<\/?strong>/g,''));
}

TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[1.48,1.38],[4.76,0.10]]);
TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[5.09,5.80], [3.24,5.90]]);
TestData([[1.0,1.0],[1.5,2.0],[3.0,4.0],[5.0,7.0],[3.5,5.0],[4.5,5.0],[3.5,4.5]],[[1,1],[5,7]]);
TestData([[1,1],[2,1],[4,3],[5,4]],[[1,1],[2,1]]);
TestData([[1,1,2],[2,1,3],[4,3,2],[5,4,4],[4,4,4]],[[1,1,2],[2,1,3]]);
TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[5.09,5.80], [3.24,5.90], [1.68,4.90]]); */


// ---------------------------------------------------- EOF K-Means Clustering Algorithm ----------------------------------------------------------


// ---------------------------------------------------- BOF Backpropagation Neural Network ----------------------------------------------------------
var ejs_neural = require('./supervised/ejs_neural');
var opt = {
	learning_rate:0.05,
	momentum:0.01,
	maxEpochs:500
};

/* var TT = [[5.1,3.5,1.4,0.2,0,0,1],[4.9,3,1.4,0.2,0,0,1],[7,3.2,4.7,1.4,0,1,0],[6.4,3.2,4.5,1.5,0,1,0],[6.3,3.3,6,2.5,1,0,0],[5.8,2.7,5.1,1.9,1,0,0],[5.7,2.8,4.6,1.5,0,1,0],[6.8,3,5.5,2.1,1,0,0],[6.4,3.1,5.5,1.8,1,0,0],[5.9,3.0,5.1,1.8,1,0,0]];

var nn = new ejs_neural.Neural(4,7,3,opt);
// nn.train(TT);
var data = nn.makeTrainTest(TT);
console.log(data);
console.log('Mulai training ...');
nn.train(data.data_training);
console.log('Training selesai ...');
console.log('Final nn weights ...');
console.log(nn.toJSON());
console.log('Akurasi data training : %d',nn.accuracy(data.data_training));
console.log('Akurasi data test : %d',nn.accuracy(data.data_test));
// console.log('Akurasi data : %d',nn.accuracy(TT));
console.log(nn.forward([5.1,3.5,1.4,0.2]).outputs); //001
console.log(nn.forward([5.9,3.0,5.1,1.8]).outputs); //100
console.log(nn.forward([6.4,3.2,4.5,1.5]).outputs); //010
console.log(nn.forward([5.5,2.5,4,1.3]).outputs); //010 */


// var TT = [[-1,-1,1,-1],[-1,1,1,1],[1,-1,1,1],[1,1,1,-1]]; // xor
// var TT = [[-1,-1,1,-1],[-1,1,1,-1],[1,-1,1,-1],[1,1,1,1]]; // and
var TT = [[-1,-1,1,-1],[-1,1,1,1],[1,-1,1,1],[1,1,1,1]]; // or
// var json = '{"0":-1.8570453538530167,"1":-1.0578781246699556,"2":1.6716922611300928,"3":-1.1001127357871128,"4":1.8538154659419666,"5":-1.124630964477959,"6":-1.677438527019374,"7":-1.0075326202144037,"8":-1.0230813212370369,"9":0.46272458742612566,"10":-0.6512284898272416,"11":-0.6612252388250472,"12":-0.9230821882924356,"13":0.7306611846866936,"14":-1.066699519330741,"15":-0.5033335760896673,"16":3.412458805650443,"17":1.6494378062956072,"18":3.3143332296912598,"19":-1.6840062015951618,"20":0.2616096437809812}';
var nn = new ejs_neural.Neural(3,4,1,opt);
// tes.fromJSON(json);
// nn.initWeights(ejs_neural.range(0.01,3*4+4+4*1+1,0.01))
nn.train(TT);
// var data = nn.makeTrainTest(TT);
// console.log(data);
// console.log('Mulai training ...');
// nn.train(data.data_training);
// console.log('Training selesai ...');
// console.log('--------------- Final nn weights -----------------');
// console.log(nn.toJSON());
// console.log('--------------------------------------------------');
// console.log('Akurasi data training : %d',nn.accuracy(data.data_training));
// console.log('Akurasi data test : %d',nn.accuracy(data.data_test));
console.log('Akurasi data : %d',nn.accuracy(TT));

console.log('Actual %d, prediction : %d',-1,nn.forward([-1,-1,1]).outputs[0]); // -1 or
console.log('Actual %d, prediction : %d',1,nn.forward([1,1,1]).outputs[0]); // 1 or
console.log('Actual %d, prediction : %d',1,nn.forward([-1,1,1]).outputs[0]); // 1 or
console.log('Actual %d, prediction : %d',1,nn.forward([1,-1,1]).outputs[0]); // 1 or


// ---------------------------------------------------- EOF Backpropagation Neural Network ----------------------------------------------------------









