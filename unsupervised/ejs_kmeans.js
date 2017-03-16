/* Program: K-Means Clustering Algorithm
Version : 1.0.1
Language : Javascript
Description : Penerapan K-Means Clustering pada Javascript
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

var ejs_kmeans = ejs_kmeans || { REVISION: 'ALPHA' };
(function(global){
	"use strict";

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
	}
	var randf = function(a, b){return Math.random()*(b-a)+a;}
	var randi = function(a, b){return Math.floor(Math.random()*(b-a)+a);}
	var randn = function(mu, std){return mu+gaussRandom()*std;}

	var zeros = function(n){
		if(typeof(n)==='undefined' || isNaN(n)){ return [];}
		if(typeof ArrayBuffer === 'undefined'){
			// lacking browser support
			var arr = new Array(n);
			for(var i=0;i<n;i++){arr[i]= 0;}
			return arr;
		} else {
			return new Float64Array(n);
		}
	}
	
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
	global.zeros = zeros;
	global.euclidean_distance = euclidean_distance;
	global.manhattan_distance = manhattan_distance;
	global.assert = assert;
})(ejs_kmeans);


(function(global){
	"use strict";
	
	// Data to use in clustering, d adalah data dan c adalah konstanta, bila c kosong maka c akan berisi random data, bila ada maka nilainya sama dengan c
	var Data = function (d,c){
		if(Object.prototype.toString.call(d) === '[object Array]' && typeof c === 'undefined'){
			this.d = d;
			this.c = 0.0;
			this.mData = d;
			this.mCluster = 0;
			this.size = this.mData.length;
		}else{
			this.d = d;
			this.c = c;
			this.mData = global.zeros(d);
			this.mCluster = 0;
			this.size = this.mData.length;
			if(typeof c === 'undefined'){
				for(var i=0;i<d;i++) { 
					this.mData[i] = global.randf(0.0, d);
				}
			}else{
				for(var i=0;i<d;i++) { 
					this.mData[i] = c;
				}
			}
		}
	}
	
	Data.prototype = {
		val : function(i,v){
			if(i !== undefined && v !== undefined){
				this.mData[i] = v;
			}else{
				return this.mData;
			}
		},
		get : function(i){
			return this.mData[i];
		},
		size : function(){
			return this.size;
		},
		cluster : function(cNumber){
			if(cNumber !== undefined){
				this.mCluster = cNumber;
			}else{
				return this.mCluster;
			}
		},
		cloneAndZero: function(){ return new Data(this.d, 0.0)},
		clone: function(){
			var D = new Data(this.d, 0.0);
			var n = D.size;
			for(var i=0;i<n;i++){ D.mData[i] = this.mData[i]; }
			return D;
		},
		toJSON: function(){
			var json = {}
			json.mData = this.mData;
			json.size = this.size;
			json.mCluster = this.mCluster;
			return json;
		},
		fromJSON: function(json){
			this.size = json.size;
			this.mData = global.zeros(this.size);
			this.mCluster = json.mCluster;
			
			for(var i=0;i<this.size;i++){
				this.mData[i] = json.mData[i];
			}
		}
	}
	
	global.Data = Data;
})(ejs_kmeans);


(function(global){
	"use strict";
	
	var Data = global.Data; // convenience
	
	var k_mean_cluster = function (samples){
		this.n_cluster = 0;
		this.samples = samples;
		this.t_data = samples.length;
		this.dataset = [];
		this.centroids = [];
		
		this.bigNumber = Math.pow(10,10);
		this.minimum = this.bigNumber;
		this.distance = 0.0;
		this.sampleNumber = 0;
		this.cluster = 0;
		this.isStillMoving = true;
		this.hasil = '';
	}
	
	k_mean_cluster.prototype = {
		initialize : function(iCentroid){
			if(typeof iCentroid === 'undefined'){
				var n = this.samples.length - 1;
				this.n_cluster = this.samples[global.randi(0,n)].length;
				this.hasil += 'Centroids initialized at:<br />';
				for(i=0;i<this.n_cluster;i++){
					this.centroids.push(this.samples[i]);
					this.hasil += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(" + this.centroids[i] + ")<br />";
				}
			}else{
				this.n_cluster = iCentroid.length;
				this.hasil += 'Centroids initialized at:<br />';
				for(var i=0;i<this.n_cluster;i++){
					this.centroids.push(iCentroid[i]);
					this.hasil += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(" + this.centroids[i] + ")<br />";
				}
			}
			this.hasil += "<br />";
			return this.centroids;
		},
		calculate : function(){
			var calc_centroids = function(n_cluster,dataset){
				var new_centroid = [];
				for(var i=0;i<n_cluster;i++){
					var total = [];
					for(var k=0;k<dataset[0].size;k++){
						var tmp = [];
						for(var j=0;j<dataset.length;j++){
							if(dataset[j].cluster() == i){
								tmp.push(dataset[j].val()[k]);
							}
						}
						if(tmp.length > 0){
							var totalInCluster = tmp.length;
							var calc = tmp.reduce(function(a,b){return a+b;}) / totalInCluster;
							total.push(calc);
						}
					}
					new_centroid.push(total);
				}
				return new_centroid;
			};
			
			while(this.dataset.length < this.t_data){
				this.dataset.push(new Data(this.samples[this.sampleNumber]));
				this.sampleNumber++;
			}
			
			// calculate Centroid Distance
			var euclidean_dist = [];
			for(var i=0;i<this.dataset.length;i++){
				var buff = [];
				for(var j=0;j<this.centroids.length;j++){
					buff.push(global.euclidean_distance(this.centroids[j],this.dataset[i].val()));
				}
				euclidean_dist.push(buff);
			}
			
			// Object clustering
			for(i=0;i<euclidean_dist.length;i++){
				this.minimum = this.bigNumber;
				for(j=0;j<this.n_cluster;j++){
					for(var k=0;k<euclidean_dist[i].length;k++){
						this.distance = euclidean_dist[i][k];
						if(this.distance < this.minimum){
							this.minimum = this.distance;
							this.cluster = k;
						}
					}
				}
				this.dataset[i].cluster(this.cluster);
			}
			
			// Calculate centroids
			var nCentroid = calc_centroids(this.n_cluster,this.dataset);
			for(i=0;i<this.n_cluster;i++){
				if(nCentroid[i].length > 0){
					this.centroids[i] = nCentroid[i];
				}
			}
			
			// Shifting centroids.
			while(this.isStillMoving){
				// calculate new centroids
				var c = calc_centroids(this.n_cluster,this.dataset);
				for(i=0;i<this.n_cluster;i++){
					if(c[i].length > 0){
						this.centroids[i] = c[i];
					}
				}
				
				// Assign all data to the ncentroids
				this.isStillMoving = false;
				
				euclidean_dist = [];
				for(i=0;i<this.dataset.length;i++){
					buff = [];
					for(j=0;j<this.centroids.length;j++){
						buff.push(global.euclidean_distance(this.centroids[j],this.dataset[i].val()));
					}
					euclidean_dist.push(buff);
				}
				
				for(i=0;i<euclidean_dist.length;i++){
					this.minimum = this.bigNumber;
					for(j=0;j<this.n_cluster;j++){
						for(k=0;k<euclidean_dist[i].length;k++){
							this.distance = euclidean_dist[i][k];
							if(this.distance < this.minimum){
								this.minimum = this.distance;
								this.cluster = k;
							}
						}
					}
					if(this.dataset[i].cluster() != this.cluster){
						this.dataset[i].cluster(this.cluster);
						this.isStillMoving = true;
					}else{
						this.dataset[i].cluster(this.cluster);
					}
				}
			}
		},
		result : function(){
			for(var i=0;i<this.n_cluster;i++){
				this.hasil += 'Cluster '+i+' includes:<br />';
				for(var j=0;j<this.t_data;j++){
					if(this.dataset[j].cluster() == i){
						this.hasil += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(" + this.dataset[j].val() + ")<br />";
					}
				}
				this.hasil += "<br />";
			}
			
			this.hasil += 'Centroids finalized at: <br />';
			for(i=0;i<this.n_cluster;i++){
				this.hasil += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>(" + this.centroids[i] + ")</strong><br />";
			}
			this.hasil += "<br />";
			return this.hasil;
		},
		toJSON: function(){
			var json = {}
			json.n_cluster = this.n_cluster;
			json.dataset = this.dataset;
			json.centroids = this.centroids;
			return json;
		},
		fromJSON: function(json){
			var n = json.dataset.length;
			this.n_cluster = json.n_cluster;
			this.dataset = global.zeros(n);
			this.centroids = global.zeros(this.n_cluster);
			
			for(var i=0;i<this.n_cluster;i++){
				this.centroids[i] = json.centroids[i];
			}
			
			for(var i=0;i<n;i++){
				this.dataset[i] = json.dataset[i];
			}
		}
	}
	
	global.k_mean_cluster = k_mean_cluster;
})(ejs_kmeans);

(function(lib){
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined"){
    window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(ejs_kmeans);