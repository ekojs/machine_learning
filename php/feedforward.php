<?php
/**
 * Feed Forward NN
 * Created by Eko Junaidi Salam <eko_junaidisalam@live.com>
 */

class feedforward{
    private $n_dim_input;
    private $n_dim_hidden;
    private $n_dim_output;

    private $input = [];
    private $ih_weights = [];
    private $h_bias = [];
    private $ho_weights = [];
    private $o_bias = [];
    private $output = [];

    public function __construct($n_dim_input,$n_dim_hidden,$n_dim_output) {
		$this->n_dim_input = $n_dim_input;
		$this->n_dim_hidden = $n_dim_hidden;
		$this->n_dim_output = $n_dim_output;
        $this->inputs = array_fill(0,$this->n_dim_input,0);
        $this->ih_weights = array_fill(0,$this->n_dim_input*$this->n_dim_hidden,0);
        $this->h_bias = array_fill(0,$this->n_dim_hidden,0);
        $this->ho_weights = array_fill(0,$this->n_dim_hidden*$this->n_dim_output,0);
        $this->o_bias = array_fill(0,$this->n_dim_output,0);
        $this->output = array_fill(0,$this->n_dim_output,0);
        
		$this->initWeights();
    }
    
    private function initWeights(){
        $weights = function(){
            $n_weights = ($this->n_dim_input*$this->n_dim_hidden)+$this->n_dim_hidden+($this->n_dim_hidden*$this->n_dim_output)+$this->n_dim_output;

            $scale = sqrt(1/$n_weights);
            for($i=0;$i<$n_weights;$i++){
                yield $this->randn(0.0,$scale);
            }
        };
        
        $arr_weights = iterator_to_array($weights());
        $this->ih_weights = array_map(function($v) use(&$arr_weights){return array_pop($arr_weights);},$this->ih_weights);
        $this->h_bias = array_map(function($v) use(&$arr_weights){return array_pop($arr_weights);},$this->h_bias);
        $this->ho_weights = array_map(function($v) use(&$arr_weights){return array_pop($arr_weights);},$this->ho_weights);
        $this->o_bias = array_map(function($v) use(&$arr_weights){return array_pop($arr_weights);},$this->o_bias);
    }

    public function setIHWeights($ih_weights){
        if(!is_array($ih_weights) || count($ih_weights) != count($this->ih_weights)) trigger_error("Data matrix ih_weights harus berupa array, dan berdimensi ".($this->n_dim_input*$this->n_dim_hidden),E_USER_ERROR);

        $this->ih_weights = array_map(function($v){return $v;},$ih_weights);
    }

    public function setHBias($h_bias){
        if(!is_array($h_bias) || count($h_bias) != count($this->h_bias)) trigger_error("Data matrix h_bias harus berupa array, dan berdimensi ".$this->n_dim_hidden,E_USER_ERROR);

        $this->h_bias = array_map(function($v){return $v;},$h_bias);
    }

    public function setHOWeights($ho_weights){
        if(!is_array($ho_weights) || count($ho_weights) != count($this->ho_weights)) trigger_error("Data matrix ho_weights harus berupa array, dan berdimensi ".($this->n_dim_output*$this->n_dim_hidden),E_USER_ERROR);

        $this->ho_weights = array_map(function($v){return $v;},$ho_weights);
    }

    public function setOBias($o_bias){
        if(!is_array($o_bias) || count($o_bias) != count($this->o_bias)) trigger_error("Data matrix o_bias harus berupa array, dan berdimensi ".$n_dim_output,E_USER_ERROR);

        $this->o_bias = array_map(function($v){return $v;},$o_bias);
    }

    public function getWeights(){
        return array(
            "ih_weights" => $this->ih_weights,
            "ho_weights" => $this->ho_weights,
        );
    }

    public function getBias(){
        return array(
            "h_bias" => $this->h_bias,
            "o_bias" => $this->o_bias,
        );
    }

    public function activation($x,$type="sigmoid"){
        $res = 0;
        switch($type){
            case "sigmoid":
                $res =  1 / (1+exp(-$x));
                break;
            case "softmax":
                $denom = array_reduce($x,function($i,$v){return $i+=exp($v);});
                $res = array_map(function($v) use($denom){return exp($v)/$denom;},$x);
                break;
        }
        return $res;
    }

    public function gaussRandom(){
        $return_v = false;
		$v_val = 0.0;
		if($return_v){
			$return_v = false;
			return $v_val;
		}
		$u = 2*(0.01*random_int(0,100))-1;
		$v = 2*(0.01*random_int(0,100))-1;
		$r = ($u*$u) + ($v*$v);
		if($r == 0 || $r > 1) return $this->gaussRandom();
		$c = sqrt(-2*log($r)/$r);
		$v_val = $v*$c; // cache this
		$return_v = true;
		return $u*$c;
    }

    public function randn($mu,$std){
        return $mu+$this->gaussRandom()*$std;
    }

    public function computeOutput($input){
        if(!is_array($input)) trigger_error("Data Input harus berupa array",E_USER_ERROR);
        if(count($input) !== $this->n_dim_input) trigger_error("Ukuran matrix data input tidak sama",E_USER_ERROR);
        
        $this->input = array_map(function($v){return $v;},$input);

        $h_outputs = array_fill(0,$this->n_dim_hidden,0);
        $k = 0;
        for($i=0;$i<$this->n_dim_input;$i++){
            for($j=0;$j<$this->n_dim_hidden;$j++){
                $h_outputs[$j] += ($this->input[$i]*$this->ih_weights[$k++]);
            }
        }

        $h_outputs = array_map(function($v,$k) use($h_outputs){return tanh($h_outputs[$k]+$v);},$this->h_bias,array_keys($this->h_bias));

        $outputs = array_fill(0,$this->n_dim_output,0);
        $k = 0;
        for($i=0;$i<$this->n_dim_hidden;$i++){
            for($j=0;$j<$this->n_dim_output;$j++){
                $outputs[$j] += ($h_outputs[$i]*$this->ho_weights[$k++]);
            }
        }

        $o_sums = array_map(function($v,$k) use($outputs){return $outputs[$k]+$v;},$this->o_bias,array_keys($this->o_bias));
        
        $this->output = $this->activation($o_sums,"softmax");

        return array(
            "h_outputs" => $h_outputs,
            "outputs" => $this->output
        );
    }

    public function cMatrix($r,$c,$v){
        $arr = array_fill(0,$r,$v);
        return ($c > 0?array_map(function($val) use($c,$v){return array_fill(0,$c,$v);},$arr):$arr);
    }
	
	public function showMatrix($mx){
		if(!is_array($mx)) trigger_error("Data matrix harus berupa array",E_USER_ERROR);
		
		for($i=0;$i<count($mx);$i++){
			echo ($i+1)." .)\t";
			if(is_array($mx[0])){
                for($j=0;$j<count($mx[0]);$j++){
                    echo $mx[$i][$j]."\t";
                }
            }else{
                echo $mx[$i]."\t";
            }
			echo "\n";
		}
	}
}

$ff = new feedforward(3,4,2);
// $ff->setIHWeights(range(0.01,0.12,0.01));
// $ff->setHBias(range(0.13,0.16,0.01));
// $ff->setHOWeights(array(0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24));
// $ff->setOBias(range(0.25,0.26,0.01));

// print_r($ff->getWeights());
// print_r($ff->getBias());

print_r($ff->computeOutput(array(1,2,3)));