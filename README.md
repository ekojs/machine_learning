### Machine Learning & Computational Intelligence
#### Berisi kumpulan algoritma dari Machine Learning dan Computational Intelligance

### K-Means Clustering Algorithm

Contoh menggunakan nodejs console :

```javascript
var ejs_kmeans = require('./ejs_kmeans');
function TestData(samples,centroid){
	console.log('Samples Data : %s \n','('+samples.join(') (')+')');
	var k_means = new ejs_kmeans.k_mean_cluster(samples);
	k_means.initialize(centroid);
	k_means.calculate();
	console.log(k_means.result().replace(/<br \/>/g,"\n").replace(/&nbsp;/g,' ').replace(/<\/?strong>/g,''));
}

TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[1.48,1.38],[4.76,0.10]]);
//TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[5.09,5.80], [3.24,5.90]]);
//TestData([[1.0,1.0],[1.5,2.0],[3.0,4.0],[5.0,7.0],[3.5,5.0],[4.5,5.0],[3.5,4.5]],[[1,1],[5,7]]);
//TestData([[1,1],[2,1],[4,3],[5,4]],[[1,1],[2,1]]);
//TestData([[1,1,2],[2,1,3],[4,3,2],[5,4,4],[4,4,4]],[[1,1,2],[2,1,3]]);
//TestData([[5.09,5.80], [3.24,5.90], [1.68,4.90], [1.00,3.17], [1.48,1.38], [2.91,0.20], [4.76,0.10], [6.32,1.10], [7.00,2.83], [6.52,4.62]],[[5.09,5.80], [3.24,5.90], [1.68,4.90]]);
```