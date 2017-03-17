/* Program: Artificial Neural Network Algorithm
Version : 1.2.1
Language : Javascript
Description : Penerapan ANN pada Javascript
Author : Eko Junaidi Salam
Email : eko_junaidisalam@live.com */

/* MIT License

Copyright (c) 2017 Eko Junaidi Salam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

var ejs_neural = ejs_neural || {REVISION: 'ALPHA' };
(function(global){
	"use strict";

	// Random number utilities
	var gaussRandom = function(){
		var return_v = false;
		var v_val = 0.0;
		if(return_v){
			return_v = false;
			return v_val; 
		}
		var u = 2*Math.random()-1;
		var v = 2*Math.random()-1;
		var r = u*u + v*v;
		if(r == 0 || r > 1) return gaussRandom();
		var c = Math.sqrt(-2*Math.log(r)/r);
		v_val = v*c; // cache this
		return_v = true;
		return u*c;
	};
	var randf = function(a, b){return Math.random()*(b-a)+a;};
	var randi = function(a, b){return Math.floor(Math.random()*(b-a)+a);};
	var randn = function(mu, std){return mu+gaussRandom()*std;};

	// Fisher-Yates Shuffle
	var shuffle = function(a){
		var counter = a.length;
		while (counter > 0) {
			var index = Math.floor(Math.random() * counter);
			counter--;
			var temp = a[counter];
			a[counter] = a[index];
			a[index] = temp;
		}
		return a;
	}

	var getopt = function(opt, field_name, default_value){
		if(typeof field_name === 'string'){
			// case of single string
			return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
		}else{
			var ret = default_value;
			for(var i=0;i<field_name.length;i++){
				var f = field_name[i];
				if (typeof opt[f] !== 'undefined'){
					ret = opt[f];
				}
			}
			return ret;
		}
	}

	var cMatrix = function(r,c,v){
		var result = new Array(r);
		for(var i=0;i<r;i++){
			result[i] = new Array(c);
			for(var j=0;j<c;j++){
				result[i][j] = (typeof v !== 'undefined'?v:0);
			}
		}
		
		return result;
	};

	var pad = function(pad, str, padLeft){
		if(typeof str === 'undefined') return pad;
		if(padLeft){
			return (pad + str).slice(-pad.length);
		}else{
			return (str + pad).substring(0, pad.length);
		}
	}

	var range = function(start, count,inc){
		var precise = (inc+'').split('.').map(function(v){return v.length;});
		inc = (typeof inc != 'undefined'?inc:1);
		return Array.apply(0, Array(count)).map(function(e,i){return parseFloat(start+(i*inc));});
	};

	var MaxIndex = function(a){
		var idx = 0;
		var max = a[0];
		for(var i=0;i<a.length;i++){
			if(a[i] > max){
				max = a[i];
				idx = i;
			}
		}
		return idx;
	};

	var zeros = function(n,v){
		if(typeof(n)==='undefined' || isNaN(n)){ return [];}
		if(typeof ArrayBuffer === 'undefined'){
			// lacking browser support
			var arr = new Array(n);
			for(var i=0;i<n;i++){arr[i]= (typeof v !== 'undefined'?v:0);}
			return arr;
		}else if(typeof v !== 'undefined'){
			var arr = new Array(n);
			for(var i=0;i<n;i++){arr[i]=v;}
			return arr;
		}else{
			return new Float64Array(n);
		}
	}
	
	var arrContains = function(arr, elt){
		for(var i=0,n=arr.length;i<n;i++){
			if(arr[i]===elt) return true;
		}
		return false;
	}

	var arrUnique = function(arr){
		var b = [];
		for(var i=0,n=arr.length;i<n;i++){
		if(!arrContains(b, arr[i])){
			b.push(arr[i]);
		}
		}
		return b;
	}

	var tanh = function(x){
		var y = Math.exp(2 * x);
		return (y - 1) / (y + 1);
	};

	var sigmoid = function(x){
		return 1/(1+Math.exp(-x));
	}

	var softmax = function(Osums){
		if(!Object.prototype.toString.call(Osums) === '[object Array]'){
			throw new Error('Input softmax bukan array');
		}
		var result = zeros(Osums.length);
		var max = Osums[0];
		var scale = 0.0;
		
		for(var i=0;i<Osums.length;i++){
			if(Osums[i] > max) max = Osums[i];
		}
		
		for(var i=0;i<Osums.length;i++){
			scale += Math.exp(Osums[i] - max);
		}
		
		for(var i=0;i<Osums.length;i++){
			result[i] += Math.exp(Osums[i] - max)/scale;
		}
		return result;
	};

	var softmaxNaive = function(Osums){
		var scale =0.0;
		var result = new Array(Osums.length);
		
		for(var i=0;i<Osums.length;i++){
			scale += Math.exp(Osums[i]);
		}
		
		for(var i=0;i<Osums.length;i++){
			result[i] += Math.exp(Osums[i])/scale;
		}
		return result;
	};

	// return max and min of a given non-empty array.
	var maxmin = function(w){
		if(w.length === 0){return {};} // ... ;s
		var maxv = w[0];
		var minv = w[0];
		var maxi = 0;
		var mini = 0;
		var n = w.length;
		for(var i=1;i<n;i++){
			if(w[i] > maxv){maxv = w[i]; maxi = i;} 
			if(w[i] < minv){minv = w[i]; mini = i;} 
		}
		return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
	}

	// create random permutation of numbers, in range [0...n-1]
	var randperm = function(n){
		var i = n,
		j = 0,
		temp;
		var array = [];
		for(var q=0;q<n;q++)array[q]=q;
		while (i--){
			j = Math.floor(Math.random() * (i+1));
			temp = array[i];
			array[i] = array[j];
			array[j] = temp;
		}
		return array;
	}
	
	var gaussNorm = function(x,rng){
		if(!Object.prototype.toString.call(rng) === '[object Array]'){
			throw new Error('Param ke 2 harus array');
		}
		var n = rng.length;
		var mean = rng.reduce(function(a,b){return a+b;})/n;
		var sums = 0;
		for(var i=0;i<rng.length;i++){
			sums += Math.pow(rng[i]-mean,2);
		}
		return (x-mean)/(Math.sqrt(sums/n));
	};
	
	var euclidean_distance = function(c,d){
		var dist = 0;
		if(c.length == d.length){
			for(var i=0;i<c.length;i++){
				dist += Math.pow(c[i]-d[i],2);
			}
			return Math.sqrt(dist);
		}else{
			throw new Error('Dimensi tidak sama');
		}
	};
	
	var manhattan_distance = function (c,d){
		var dist = 0;
		if(c.length == d.length){
			for(var i=0;i<c.length;i++){
				dist += Math.abs(c[i]-d[i]);
			}
			return dist;
		}else{
			throw new Error('Dimensi tidak sama');
		}
	}

	function assert(condition, message){
		if (!condition){
			message = message || "Assertion gagal !!!";
			if (typeof Error !== "undefined"){
				throw new Error(message);
			}
			throw message; // Fallback
		}
	}

	global.randf = randf;
	global.randi = randi;
	global.randn = randn;
	global.shuffle = shuffle;
	global.getopt = getopt;
	global.cMatrix = cMatrix;
	global.pad = pad;
	global.range = range;
	global.MaxIndex = MaxIndex;
	global.zeros = zeros;
	global.arrContains = arrContains;
	global.arrUnique = arrUnique;
	global.tanh = tanh;
	global.sigmoid = sigmoid;
	global.softmax = softmax;
	global.softmaxNaive = softmaxNaive;
	global.maxmin = maxmin;
	global.randperm = randperm;
	global.gaussNorm = gaussNorm;
	global.euclidean_distance = euclidean_distance;
	global.manhattan_distance = manhattan_distance;
	global.assert = assert;
})(ejs_neural);


(function(global){
	"use strict";
	
	var Neural = function(ninput,nhidden,noutput,opt){
		var opt = opt || {};
		
		this.input = ninput;
		this.hidden = nhidden;
		this.output = noutput;
		
		// Feed Forward
		this.inputs = global.zeros(this.input);
		this.IHweights = global.cMatrix(this.input,this.output);
		this.Hbias = global.zeros(this.hidden);
		this.Houtput = global.zeros(this.hidden);
		this.HOweights = global.cMatrix(this.hidden,this.output);
		this.Obias = global.zeros(this.output);
		this.outputs = global.zeros(this.output);
		
		// Backpropagation
		this.learnRate = global.getopt(opt, 'learning_rate', 0.05);
		this.momentum = global.getopt(opt, 'momentum', 0.01);
		this.maxEpochs = global.getopt(opt, 'maxEpochs', 1000);
		
		this.Ograds = global.zeros(this.output);
		this.Hgrads = global.zeros(this.hidden);
		this.IHPrevWeights = global.cMatrix(this.input,this.hidden,0.011);
		this.HPrevBiases = global.zeros(this.hidden,0.011);
		this.HOPrevWeights = global.cMatrix(this.hidden,this.output,0.011);
		this.OPrevBiases = global.zeros(this.output,0.011);
		this.initWeights();
	}
	
	Neural.prototype = {
		initWeights: function(weights){
			var numWeights = (this.input * this.hidden) + this.hidden + (this.hidden * this.output) + this.output;
			var w = (typeof weights != 'undefined'?weights:global.zeros(numWeights));
			
			if(typeof weights === 'undefined'){
				var scale = Math.sqrt(1.0/(numWeights));
				for(var i=0;i<numWeights;i++) { 
					w[i] = global.randn(0.0, scale);
				}
			}
			
			if(numWeights != w.length){
				throw new Error('Jumlah weights tidak sama');
			}
			var k = 0;
			for(var i=0;i<this.input;i++){
				for(var j=0;j<this.hidden;j++){
					this.IHweights[i][j] = w[k++];
				}
			}
			for(var i=0;i<this.hidden;i++){
				this.Hbias[i] = w[k++];
			}
			for(var i=0;i<this.hidden;i++){
				for(var j=0;j<this.output;j++){
					this.HOweights[i][j] = w[k++];
				}
			}
			for(var i=0;i<this.output;i++){
				this.Obias[i] = w[k++];
			}
			return {
				IHweights: this.IHweights,
				Hbias: this.Hbias,
				HOweights: this.HOweights,
				Obias: this.Obias
			};
		},
		
		findWeights: function(tVal,xVal){
			var epoch = 0;
			while(epoch <= this.maxEpochs){
				var out = this.forward(xval);
				this.backward(tVal);
				if(epoch % 100 == 0){
					console.log('epoch = '+pad('0000',epoch,true)+' curr outputs = '+out.outputs);
				}
				epoch++;
			}
		},
		
		getWeights: function(){
			var numWeights = (this.input * this.hidden) + this.hidden + (this.hidden * this.output) + this.output;
			var result = global.zeros(numWeights);
			var k = 0;
			for(var i=0;i<this.input;i++){
				for(var j=0;j<this.hidden;j++){
					result[k++] = this.IHweights[i][j];
				}
			}
			for(var i=0;i<this.hidden;i++){
				result[k++] = this.Hbias[i];
			}
			for(var i=0;i<this.hidden;i++){
				for(var j=0;j<this.output;j++){
					result[k++] = this.HOweights[i][j];
				}
			}
			for(var i=0;i<this.output;i++){
				result[k++] = this.Obias[i];
			}
			return result;
		},
		
		toJSON: function(){
			return JSON.stringify(this.getWeights());
		},
		
		fromJSON: function(json){
			var obj = JSON.parse(json);
			var result = [];
			for(d in obj){
				result.push(obj[d]);
			}
			this.initWeights(result);
		},
		
		forward: function(input){
			if(input.length != this.input){
				throw new Error('Jumlah data tidak sama');
			}
			var Hsums = global.zeros(this.hidden);
			var Osums = global.zeros(this.output);
			
			for(var i=0;i<input.length;i++){
				this.inputs[i] = input[i];
			}
			for(var j=0;j<this.hidden;j++){
				for(var i=0;i<this.input;i++){
					Hsums[j] += this.inputs[i] * this.IHweights[i][j];
				}
			}
			for(var i=0;i<this.hidden;i++){
				Hsums[i] += this.Hbias[i];
			}
			// console.log('Pre Activation hidden sums :');
			// console.log(Hsums);
			
			for(var i=0;i<this.hidden;i++){
				this.Houtput[i] = global.tanh(Hsums[i]);
			}
			// console.log('Hidden output :');
			// console.log(this.Houtput);
			
			for(var j=0;j<this.output;j++){
				for(var i=0;i<this.hidden;i++){
					Osums[j] += this.Houtput[i] * this.HOweights[i][j];
				}
			}
			for(var i=0;i<this.output;i++){
				Osums[i] += this.Obias[i];
			}
			// console.log('Pre Activation output sums :');
			// console.log(Osums);
			
			var softOut = global.softmax(Osums);
			// console.log('Osums : %d, softmax : %d',Osums,softOut);
			// console.log(Osums);
			
			for(var i=0;i<this.outputs.length;i++){
				// this.outputs[i] = softOut[i];
				this.outputs[i] = (this.outputs.length > 1?softOut[i]:global.sigmoid(Osums[i]));
			}
			
			return {
				inputs: this.inputs,
				Hsums: Hsums,
				Houtput: this.Houtput,
				Osums: Osums,
				outputs: this.outputs
			};
		},
		
		backward: function(tVal){
			if(tVal.length !== this.output){
				throw new Error('Panjang data tidak sama dengan panjang output !!!');
			}
			// 1. Compute output gradients assume using softmax
			for(var i=0;i<this.Ograds.length;i++){
				// Turunan dari softmax adalah y(1-y)
				this.Ograds[i] = ((1-this.outputs[i])*this.outputs[i]) * (tVal[i] - this.outputs[i]);
			}
			// 2. Compute hidden gradients, assume using tanh
			for(var i=0;i<this.Hgrads.length;i++){
				var derivative = (1-this.Houtput[i]) * (1+this.Houtput[i]);
				var sum = 0;
				for(var j=0;j<this.output;j++){
					sum += this.Ograds[j]*this.HOweights[i][j];
				}
				this.Hgrads[i] = derivative*sum;
			}
			// 3. Update input to hidden weights
			for(var i=0;i<this.IHweights.length;i++){
				for(var j=0;j<this.IHweights[i].length;j++){
					var delta = this.learnRate*this.Hgrads[j] * this.inputs[i];
					this.IHweights[i][j] += delta;
					// Update
					this.IHweights[i][j] += this.momentum * this.IHPrevWeights[i][j];
					// Add momentum factor
					this.IHPrevWeights[i][j] = delta;
				}
			}
			// 4. Update Hidden Biasses
			for(var i=0;i<this.Hbias.length;i++){
				var delta = this.learnRate * this.Hgrads[i] * 1.0;
				this.Hbias[i] += delta;
				this.Hbias[i] += this.momentum * this.HPrevBiases[i];
				this.HPrevBiases[i] = delta;
			}
			// 5. Update hidden to output weights
			for(var i=0;i<this.HOweights.length;i++){
				for(var j=0;j<this.HOweights[i].length;j++){
					var delta = this.learnRate * this.Ograds[j] * this.Houtput[i];
					this.HOweights[i][j] += delta;
					this.HOweights[i][j] += this.momentum * this.HOPrevWeights[i][j];
					// console.log('Update hidden to output weights i=%d j=%d delta=%d HOPrevWeights=%d : %d',i,j,delta,this.HOPrevWeights[i][j],this.HOweights[i][j]);
					this.HOPrevWeights[i][j] = delta;
				}
			}
			// 6. Update output biases
			for(var i=0;i<this.Obias.length;i++){
				var delta = this.learnRate*this.Ograds[i] * 1.0;
				this.Obias[i] += delta;
				this.Obias[i] += this.momentum * this.OPrevBiases[i];
				this.OPrevBiases[i] = delta;
			}
		},
		
		MeanSquaredError: function(trainData){
			var sumSquaredError = 0.0;
			var xval = new Array(this.input);
			var tval = new Array(this.output);
			
			for(var i=0;i<trainData.length;i++){
				xval = trainData[i].slice(0,this.input);
				tval = trainData[i].slice(this.input,trainData[i].length);
				var yval = this.forward(xval).outputs;
				
				for(var j=0;j<this.output;j++){
					var err = tval[j] - yval[j];
					sumSquaredError += Math.pow(err,2);
				}
			}
			return sumSquaredError/trainData.length;
		},
		
		makeTrainTest: function(allData){
			var totRows = allData.length;
			var cols = allData[0].length;
			var trainRows = Math.round(totRows * 0.70);
			var testRows = totRows - trainRows;
			var trainData = new Array(trainRows);
			var testData = new Array(testRows);
			var copied = new Array(allData.length);
			
			for(var i=0;i<copied.length;i++){
				copied[i] = allData[i];
			}
			
			copied = global.shuffle(copied);
			
			for(var i=0;i<trainRows;i++){
				trainData[i] = new Array(cols);
				for(var j=0;j<cols;j++){
					trainData[i][j] = copied[i][j];
				}
			}
			
			for(var i=0;i<testRows;i++){
				testData[i] = new Array(cols);
				for(var j=0;j<cols;j++){
					testData[i][j] = copied[i+trainRows][j];
				}
			}
			
			return {
				data_training: trainData,
				data_test: testData
			};
		},
		
		train: function(trainData){
			var epoch = 0;
			var xval = new Array(this.input);
			var tval = new Array(this.output);
			// var seq = global.range(0,trainData.length);
			
			while(epoch < this.maxEpochs){
				var mse = this.MeanSquaredError(trainData);
				
				if(epoch % 100 == 0){
					console.log('epoch = '+global.pad('0000',epoch,true)+' Err outputs = '+mse);
				}
				
				if(mse < 0.04) break;
				
				// seq = global.shuffle(seq);
				// console.log(seq);
				for(var i=0;i<trainData.length;i++){
					// xval = trainData[seq[i]].slice(0,this.input);
					// tval = trainData[seq[i]].slice(this.input,trainData[i].length);
					xval = trainData[i].slice(0,this.input);
					tval = trainData[i].slice(this.input,trainData[i].length);
					this.forward(xval);
					this.backward(tval);
				}
				epoch++;
			}
			
			return {
				IHweights: this.IHweights,
				Hbias: this.Hbias,
				HOweights: this.HOweights,
				Obias: this.Obias
			}
		},
		
		accuracy: function(testData){
			var correct = 0;
			var wrong = 0;
			var xval = new Array(this.input);
			var tval = new Array(this.output);
			var yval = new Array(this.output);
			
			for(var i=0;i<testData.length;i++){
				xval = testData[i].slice(0,this.input);
				tval = testData[i].slice(this.input,testData[i].length);
				yval = this.forward(xval).outputs;
				var maxIndex = (this.output.length > 1?global.MaxIndex(yval):yval[0]);
				var idx = (this.output.length > 1?global.MaxIndex(tval):tval[0]);
				
				// console.log('--------- BOF Accuracy --------');
				// console.log(testData);
				// console.log(xval);
				// console.log(tval);
				// console.log(yval);
				// console.log('Index y : %d, Index t : %d, total data -> %d',maxIndex,idx,testData.length);
				// console.log('--------- EOF Accuracy --------');
				
				if(this.output.length > 1){
					if(maxIndex == idx){
						correct++;
					}else{
						wrong++;
					}
				}else{
					maxIndex = Math.round(maxIndex);
					idx = (idx > 0?1:0);
					if(maxIndex == idx){
						correct++;
					}else{
						wrong++;
					}
				}
			}
			
			return (correct * 1.0) / (correct + wrong);
		}
	};
	
	global.Neural = Neural;
})(ejs_neural);


(function(lib){
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined"){
    window.jsfeat = lib; // in ordinary browser attach library to window
  }else{
    module.exports = lib; // in nodejs
  }
})(ejs_neural);