<?php
/**
	Class Feed Forward NN
	Created by Eko Junaidi Salam <eko_junaidisalam@live.com>
**/
class feedforward{
	public $numInput = array();
	public $numHidden = array();
	public $numOutput = array();
	
	public $inputs = array();
	public $ihweights = array();
	public $hbias = array();
	public $houtputs = array();
	public $howeights = array();
	public $obias = array();
	public $outputs = array();
	
	public function __construct($numInput,$numHidden,$numOutput) {
		$this->numInput = $numInput;
		$this->numHidden = $numHidden;
		$this->numOutput = $numOutput;
		$this->inputs = array_fill(0,$numInput,0);
		$this->ihweights = $this->cMatrix($numInput,$numHidden,0);
		$this->hbias = array_fill(0,$numHidden,0);
		$this->houtputs = array_fill(0,$numHidden,0);
		$this->howeights = $this->cMatrix($numHidden,$numOutput,0);
		$this->obias = array_fill(0,$numOutput,0);
		$this->outputs = array_fill(0,$numOutput,0);
	}
	
	public function setWeights($w){
		$numWeigths = ($this->numInput*$this->numHidden)+$this->numHidden+($this->numHidden*$this->numOutput)+$this->numOutput;
		if(!is_array($w)) trigger_error("Data Weights harus berupa array",E_USER_ERROR);
		if(count($w) !== $numWeigths) trigger_error("Ukuran matrix data weights tidak sama",E_USER_ERROR);
		
		$k=0;
		for($i=0;$i<$this->numInput;$i++){
			for($j=0;$j<$this->numHidden;$j++){
				$this->ihweights[$i][$j] = $w[$k++];
			}
		}
		for($j=0;$j<$this->numHidden;$j++){
			$this->hbias[$j] = $w[$k++];
		}
		for($i=0;$i<$this->numHidden;$i++){
			for($j=0;$j<$this->numOutput;$j++){
				$this->howeights[$i][$j] = $w[$k++];
			}
		}
		for($j=0;$j<$this->numOutput;$j++){
			$this->obias[$j] = $w[$k++];
		}
		return $this;
	}
	
	public function computeOutput($xvalues){
		if(!is_array($xvalues)) trigger_error("Data Input harus berupa array",E_USER_ERROR);
		if(count($xvalues) !== $this->numInput) trigger_error("Ukuran matrix data input tidak sama",E_USER_ERROR);
		
		for($i=0;$i<$this->numInput;$i++){
			$this->inputs[$i] = $xvalues[$i];
		}
		
		$hsums = array_fill(0,$this->numHidden,0);
		$osums = array_fill(0,$this->numOutput,0);
		
		for($j=0;$j<$this->numHidden;$j++){
			for($i=0;$i<$this->numInput;$i++){
				$hsums[$j] += $this->inputs[$i] * $this->ihweights[$i][$j];
			}
		}
		for($i=0;$i<$this->numHidden;$i++){
			$hsums[$i] += $this->hbias[$i];
			// $this->houtputs[$i] = tanh($hsums[$i]);
		}
		echo "Pre-activation Hidden Sums :\n";
		echo implode(",",$hsums)."\n\n";
		
		for($i=0;$i<$this->numHidden;$i++){
			// $hsums[$i] += $this->hbias[$i];
			$this->houtputs[$i] = tanh($hsums[$i]);
		}
		echo "Hidden Outputs :\n";
		echo implode(",",$this->houtputs)."\n\n";
		
		for($j=0;$j<$this->numOutput;$j++){
			for($i=0;$i<$this->numHidden;$i++){
				$osums[$j] += $this->houtputs[$i] * $this->howeights[$i][$j];
			}
		}
		
		for($i=0;$i<$this->numOutput;$i++){
			$osums[$i] += $this->obias[$i];
		}
		echo "Pre-activation Output Sums :\n";
		echo implode(",",$osums)."\n\n";
		
		$softOut = $this->softmax($osums);
		for($i=0;$i<$this->numOutput;$i++){
			$this->outputs[$i] = $softOut[$i];
		}
		
		return $this->outputs;
	}
	
	private function softmax($val){
		if(!is_array($val)) trigger_error("Data harus berupa array",E_USER_ERROR);
		$max = $val[0];
		for($i=0;$i<count($val);$i++){
			if($max < $val[$i]) $max = $val[$i];
		}
		
		$scale = 0;
		for($i=0;$i<count($val);$i++){
			$scale += exp($val[$i] - $max);
		}
		
		$result = array_fill(0,count($val),0);
		for($i=0;$i<count($val);$i++){
			$result[$i] = exp($val[$i] - $max) / $scale;
		}
		return $result;
	}
	
	private function softmaxNaive($val){
		if(!is_array($val)) trigger_error("Data harus berupa array",E_USER_ERROR);
		$denom = 0;
		for($i=0;$i<count($val);$i++){
			$denom += exp($val[$i]);
		}
		
		$result = array_fill(0,count($val),0);
		for($i=0;$i<count($val);$i++){
			$result[$i] = exp($val[$i]) / $denom;
		}
		return $result;
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

$a = new feedforward(3,4,2);
$weights = range(0.01,0.26,0.01);

$a->setWeights($weights);
$res = $a->computeOutput(array(1,2,3));

echo "Outputs : \n";
echo implode(",",$res)."\n\n";

exit;
?>