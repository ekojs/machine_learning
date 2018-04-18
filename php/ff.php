<?php
/**
	Class Feed Forward NN
	Created by Eko Junaidi Salam <eko_junaidisalam@live.com>
**/
class feedforward{
	public $numInput;
	public $numHidden;
	public $numOutput;
	public $arrInput;
	
	public $inputs = array();
	public $ihWeights = array();
	public $hBias = array();
	public $hOutputs = array();
	public $hoWeights = array();
	public $oBias = array();
	public $outputs = array();
	
	public function __construct($arrInput){
		if(!is_array($arrInput)) trigger_error("Data input harus array, misal \$in = array(4,3,2); //input,hidden,output",E_USER_ERROR);
		
		$this->arrInput = $arrInput;
		$this->numInput = $arrInput[0];
		$this->numHidden = $arrInput[1];
		$this->numOutput = $arrInput[count($arrInput)-1];
		
		$this->inputs = array_fill(0,$this->numInput,0);
		$this->ihWeights = $this->cMatrix($this->numInput,$this->numHidden,0);
		$this->hBias = array_fill(0,$this->numHidden,0);
		$this->hOutputs = array_fill(0,$this->numHidden,0);
		$this->hoWeights = $this->cMatrix($this->numHidden,$this->numOutput,0);
		$this->oBias = array_fill(0,$this->numOutput,0);
		$this->outputs = array_fill(0,$this->numOutput,0);
	}
	
	public function setWeights($w){
		$sizeW = 0;
		for($i=1;$i<count($this->arrInput);$i++){
			$sizeW += $this->arrInput[$i]*$this->arrInput[$i-1]+$this->arrInput[$i];
		}
		
		if(!is_array($w)) trigger_error("Data weights harus array !!!",E_USER_ERROR);
		if(count($w) !== $sizeW) trigger_error("Data weights harus array !!!",E_USER_ERROR);
		
		$k = 0;
		for($i=0;$i<$this->numInput;$i++){
			for($j=0;$j<$this->numHidden;$j++){
				$this->ihWeights[$i][$j] = $w[$k++];
			}
		}
		for($i=0;$i<$this->numHidden;$i++){
			$this->hBias[$i] = $w[$k++];
		}
		for($i=0;$i<$this->numHidden;$i++){
			for($j=0;$j<$this->numOutput;$j++){
				$this->hoWeights[$i][$j] = $w[$k++];
			}
		}
		for($i=0;$i<$this->numOutput;$i++){
			$this->oBias[$i] = $w[$k++];
		}
		return $this;
	}
	
	public function computeOutputs($data){
		if(!is_array($data)) trigger_error("Data harus berupa array !!!",E_USER_ERROR);
		if(count($data) !== $this->numInput) trigger_error("Jumlah data tidak sesuai input !!!",E_USER_ERROR);
		
		for($i=0;$i<$this->numInput;$i++){
			$this->inputs[$i] = $data[$i];
		}
		
		$hSums = array_fill(0,$this->numHidden,0);
		$oSums = array_fill(0,$this->numOutput,0);
		for($j=0;$j<$this->numHidden;$j++){
			for($i=0;$i<$this->numInput;$i++){
				$hSums[$j] += $this->inputs[$i]*$this->ihWeights[$i][$j];
			}
			$hSums[$j] += $this->hBias[$j];
			$this->hOutputs[$j] = tanh($hSums[$j]);
		}
		
		for($j=0;$j<$this->numOutput;$j++){
			for($i=0;$i<$this->numHidden;$i++){
				$oSums[$j] += $this->hOutputs[$i]*$this->hoWeights[$i][$j];
			}
			$oSums[$j] += $this->oBias[$j];
		}
		
		$softOut = $this->softmax($oSums);
		for($i=0;$i<count($softOut);$i++){
			$this->outputs[$i] = $softOut[$i];
		}
		return $this;
	}
	
	public function softmax($val){
		if(!is_array($val)) trigger_error("Data harus berupa array !!!",E_USER_ERROR);
		
		$max = 0;
		for($i=0;$i<count($val);$i++){
			if($max < $val[$i]) $max = $val[$i];
		}
		
		$scale = 0;
		for($i=0;$i<count($val);$i++){
			$scale += exp($val[$i]-$max);
		}
		
		$result = array_fill(0,count($val),0);
		for($i=0;$i<count($val);$i++){
			$result[$i] = exp($val[$i]-$max) / $scale;
		}
		return $result;
	}
	
	public function softmaxNaive($val){
		if(!is_array($val)) trigger_error("Data harus berupa array !!!",E_USER_ERROR);
		
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
		$arr = array_fill(0,$r,0);
		for($i=0;$i<count($arr);$i++){
			$arr[$i] = array_fill(0,$c,$v);
		}
		return $arr;
	}
}

/* $in = array(3,4,4,2); // 26, 46
$sum = 0;
for($i=1;$i<count($in);$i++){
	$sum += $in[$i]*$in[$i-1]+$in[$i];
}
echo "Jumlah weights : ".$sum."\n";

$weights = range(0.01,0.46,0.01);

$numInput = $numHidden = $numOutput = 0;

$numInput = $in[0];
$numHidden = count($in)-2;
$numOutput = $in[count($in)-1];

$iWeights = array();
$hWeights = array_fill(0,$numHidden,0);
$hBias = array_fill(0,$numHidden,0);
$oWeights = array();

$k = 0;
for($i=1;$i<count($in);$i++){
	for(){
		
	}
}

exit; */

$ff = new feedforward(array(3,4,2));
$w = range(0.01,0.26,0.01);
$ff->setWeights($w);
$out = $ff->computeOutputs(array(1,2,3));

echo implode(",",$out->outputs)."\n\n";

// print_r($ff->softmax(array(0.6908,0.7228)));
// print_r($ff->softmaxNaive(array(0.6908,0.7228)));

exit;

?>