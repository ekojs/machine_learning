<?php
/**
	Modul : Backpropagation NN
	Created By Eko Junaidi Salam <eko_junaidisalam@live.com>
**/

class backprop{
	public $arrInput = array();
	public $numInput;
	public $numHidden;
	public $numOutput;
	
	public $inputs = array();
	public $weights = array();
	public $ihWeights = array();
	public $hBias = array();
	public $hOutputs = array();
	public $hoWeights = array();
	public $oBias = array();
	public $outputs = array();

	public $oGrads = array();
	public $hGrads = array();


	public function __construct($arrInput){
		if(!is_array($arrInput)) trigger_error("Data Input harus berupa array !!!",E_USER_ERROR);

		$this->arrInput = $arrInput;
		$this->numInput = $arrInput[0];
		$this->numHidden = $arrInput[count($arrInput)-2];
		$this->numOutput = $arrInput[count($arrInput)-1];
	}	

	public function cMatrix($c,$r,$v){
		$arr = array_fill(0,$c,0);
		for($i=0;$i<$c;$i++){
			$arr[$i] = array_fill(0,$r,$v);
		}
		return $arr;
	}

	public function showMatrix(){

	}
}

$a = new backprop(array(3,4,2));
print_r($a->cMatrix(3,4,0));
exit;
?>
