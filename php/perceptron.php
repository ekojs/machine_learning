<?php
/**
	Created by Eko Junaidi Salam <eko_junaidisalam@live.com>
**/
class perceptron{
	private $numInput;
	private $inputs = array();
	private $weights = array();
	private $bias;
	private $output;
	
	public function __construct($numInput) {
		$this->numInput = $numInput;
		$this->inputs = array_fill(0,$numInput,0);
		$this->weights = array_fill(0,$numInput,0);
		$this->bias = 0;
		// $this->weights = array(0.124,-0.001);
		// $this->bias = -0.22;
		// $this->weights = array(0.038,0.038);
		// $this->bias = -0.152;
		$this->initWeights();
	}
	
	private function initWeights(){
		$this->weights = array_map(function($x){
			return (0.01*rand(0,100));
		},$this->weights);
		$this->bias = (0.01*rand(0,100));
		return $this;
	}
	
	public function computeOutput($xvalues){
		if(!is_array($xvalues)) trigger_error("Data Input harus berupa array",E_USER_ERROR);
		if(count($xvalues) !== $this->numInput) trigger_error("Ukuran matrix data input tidak sama",E_USER_ERROR);
		
		for($i=0;$i<$this->numInput;$i++){
			$this->inputs[$i] = $xvalues[$i];
		}
		
		$sums = 0;
		for($i=0;$i<$this->numInput;$i++){
			$sums += $this->inputs[$i]*$this->weights[$i];
			$sums += $this->bias;
		}
		
		$result = $this->activation($sums);
		$this->output = $result;
		return $result;
	}
	
	private function activation($v){
		return ($v >= 0?1:-1);
	}
	
	public function trainData($trainData,$learningRate,$maxEpochs){
		if(!is_array($trainData)) trigger_error("Data training harus berupa array",E_USER_ERROR);
		
		$xvalues = array_fill(0,$this->numInput,0);
		
		$epoch = 0;
		$error = $correct = 0;
		$stillLearn = true;
		while($epoch <= $maxEpochs){
		// while($stillLearn){
			for($i=0;$i<count($trainData);$i++){
				for($j=0;$j<count($xvalues);$j++){
					$xvalues[$j] = $trainData[$i][$j];
				}
				$desiredOut = $trainData[$i][$this->numInput];
				$computed = $this->computeOutput($xvalues);
				if($computed == $desiredOut){
					$correct++;
				}else{
					$error++;
				}
				$this->updateWeights($computed,$desiredOut,$learningRate);
			}
			
			$accurate = $correct/($correct+$error);
			
			if($epoch % 10 == 0){
				echo "Accurate : ".$accurate.", Epoch : ".$epoch.", Weigths : ";
				for($i=0;$i<count($this->weights);$i++){
					echo $this->weights[$i]."\t";
				}
				echo "Bias : ".$this->bias."\n";
			}
			
			if($accurate > 0.8){
				$stillLearn = false;
				echo "Accurate : ".$accurate.", Epoch : ".$epoch."\n";
				return array(
					"weights" => $this->weights,
					"bias" => $this->bias
				);
			}
			
			$error = $correct = 0;
			$epoch++;
		}
		
		return array(
			"weights" => $this->weights,
			"bias" => $this->bias
		);
	}
	
	private function updateWeights($computed,$desiredOut,$learningRate){
		if($computed == $desiredOut) return $this;
		$delta = $computed - $desiredOut;
		for($i=0;$i<count($this->weights);$i++){
			if($computed > $desiredOut && $this->inputs[$i] >= 0){
				if($this->weights[$i] >=0){
					$this->weights[$i] -= ($learningRate * $delta * $this->inputs[$i]);
				}else{
					$this->weights[$i] += ($learningRate * $delta * $this->inputs[$i]);
				}
			}else if($computed > $desiredOut && $this->inputs[$i] < 0){
				if($this->weights[$i] >=0){
					$this->weights[$i] -= ($learningRate * $delta * $this->inputs[$i]);
				}else{
					$this->weights[$i] += ($learningRate * $delta * $this->inputs[$i]);
				}
			}else if($computed < $desiredOut && $this->inputs[$i] >= 0){
				if($this->weights[$i] >=0){
					$this->weights[$i] -= ($learningRate * $delta * $this->inputs[$i]);
				}else{
					$this->weights[$i] += ($learningRate * $delta * $this->inputs[$i]);
				}
			}else if($computed < $desiredOut && $this->inputs[$i] < 0){
				if($this->weights[$i] >=0){
					$this->weights[$i] -= ($learningRate * $delta * $this->inputs[$i]);
				}else{
					$this->weights[$i] += ($learningRate * $delta * $this->inputs[$i]);
				}
			}
			
			if($computed > $desiredOut){
				$this->bias -= ($learningRate * $delta);
			}else{
				$this->bias += ($learningRate * $delta);
			}
		}
		return $this;
	}
	
	public function cMatrix($r,$c,$v){
		$arr = array_fill(0,$r,$v);
		for($i=0;$i<$r;$i++){
			$arr[$i] = array_fill(0,$c,$v);
		}
		return $arr;
	}
	
	public function showMatrix($mx){
		if(!is_array($mx)) trigger_error("Data matrix harus berupa array",E_USER_ERROR);
		
		for($i=0;$i<count($mx);$i++){
			echo ($i+1)." .)\t";
			for($j=0;$j<count($mx[0]);$j++){
				echo $mx[$i][$j]."\t";
			}
			echo "\n";
		}
	}
}

$a = new perceptron(2);
$trainData = array(
	array(1.5,2,-1),
	array(2,3.5,-1),
	array(3,5,-1),
	array(3.5,2.5,-1),
	array(4.5,5,1),
	array(5,7,1),
	array(5.5,8,1),
	array(6,6,1)
);
$newData1 = array(
	array(1.5,2),
	array(2,3.5),
	array(3,5),
	array(3.5,2.5),
	array(4.5,5),
	array(5,7),
	array(5.5,8),
	array(6,6)
);
$newData = array(
	array(3,4),  // 0
	array(0,1),  // 0
	array(2,5),  // 0
	array(5,6),  // 1
	array(9,9),  // 1
	array(4,6) ,  // 1
	array(6,7)  // 1
);
$learningRate = 0.001;
$maxEpochs = (!empty($argv[1])?intval($argv[1]):100);
echo "Data Training : \n";
$a->showMatrix($trainData);
sprintf("Learning Rate : %s, Max Epochs : %s\n",$learningRate,$maxEpochs);
echo "Start Training ....\n";
$dataT = $a->trainData($trainData,$learningRate,$maxEpochs);
echo "Training Completed....\n";
echo "Best Weights and Bias Found....\n";
print_r($dataT);

echo "Prediction for new data....\n";
for($i=0;$i<count($newData);$i++){
	echo ($i+1)." .)\t";
	for($j=0;$j<count($newData[0]);$j++){
		echo $newData[$i][$j]."\t";
	}
	echo $a->computeOutput($newData[$i])."\n";
}
?>