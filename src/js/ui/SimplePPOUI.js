class SimpleUI{
    constructor(opt){
        this.params_setter = params_setter.bind(this);
        let default_options = {
            "nn_config": [128, 128],
            "policy_nn": null,
            "parent": null,
            "download_weights_button_text": "Download agent weights",
            "set_weights_button_text": "Set Agent",
            "worker": null
        }
        this.params_setter(default_options, opt);
        this.initDownloadWeightsButton();
        this.initSetWeightsButton();
        this.initPolicyWeightsPathInput();
        this.initValueWeightsPathInput();
        
    }
    
    initDownloadWeightsButton(){
        if(document){
            this.downloadWeightsButton = document.createElement("Button");
            this.downloadWeightsButton.addEventListener("click", this.downloadWeights.bind(this));
            this.downloadWeightsButton.style.left = "0px";
            this.downloadWeightsButton.style.top = "50px";
            this.downloadWeightsButton.style.width = "100px";
            this.downloadWeightsButton.style.height = "40px";            
            this.downloadWeightsButton.id = "ppo_ui_download_weights";
            this.downloadWeightsButton.style.zIndex = 10;
            this.downloadWeightsButton.style.position = 'absolute';
            this.downloadWeightsButton.textContent = this.download_weights_button_text;
            
            if(this.parent){
                this.parent.appendChild(this.downloadWeightsButton);
            } else {
                document.body.appendChild(this.downloadWeightsButton);
                this.parent = document.body;
            }
        }
    }

    initSetWeightsButton(){
        if(document){
            this.setWeightsButton = document.createElement("Button");
            this.setWeightsButton.addEventListener("click", this.setWeights.bind(this));
            this.setWeightsButton.style.left = "0px";
            this.setWeightsButton.style.top = "150px";
            this.setWeightsButton.style.width = "100px";
            this.setWeightsButton.style.height = "40px";            
            this.setWeightsButton.id = "ppo_ui_set_weights_button";
            this.setWeightsButton.style.zIndex = 10;
            this.setWeightsButton.style.position = 'absolute';
            this.setWeightsButton.textContent = this.set_weights_button_text;

            if(this.parent){
                this.parent.appendChild(this.setWeightsButton);
            } else {
                document.body.appendChild(this.setWeightsButton);
                this.parent = document.body;
            }
        }
    }

    initPolicyWeightsPathInput(){
        if(document){
            this.policyWeightsPath = document.createElement("input");
            this.policyWeightsPath.type = "text";
            this.policyWeightsPath.style.left = "0px";
            this.policyWeightsPath.style.top = "100px";          
            this.policyWeightsPath.id = "ppo_ui_set_policy_weight_input";
            this.policyWeightsPath.style.zIndex = 10;
            this.policyWeightsPath.style.position = 'absolute';

            if(this.parent){
                this.parent.appendChild(this.policyWeightsPath);
            } else {
                document.body.appendChild(this.policyWeightsPath);
                this.parent = document.body;
            }
        }
    }

    initValueWeightsPathInput(){
        if(document){
            this.valueWeightsPath = document.createElement("input");
            this.valueWeightsPath.type = "text";
            this.valueWeightsPath.style.left = "0px";
            this.valueWeightsPath.style.top = "130px";          
            this.valueWeightsPath.id = "ppo_ui_set_value_weight_input";
            this.valueWeightsPath.style.zIndex = 10;
            this.valueWeightsPath.style.position = 'absolute';

            if(this.parent){
                this.parent.appendChild(this.valueWeightsPath);
            } else {
                document.body.appendChild(this.valueWeightsPath);
                this.parent = document.body;
            }
        }
    }


    setWeights(){
        let policy_path = window.location.origin + "/" + this.policyWeightsPath.value;
        let value_path = window.location.origin + "/" + this.valueWeightsPath.value;
        this.policy_nn = tf.loadLayersModel(policy_path);    
        this.worker.postMessage({
            msg_type: "set_policy_weights",
            policy_weights_path: policy_path,
            value_weights_path: value_path,
        });
    }

    downloadWeights(){
        this.worker.postMessage({
            msg_type: "get_policy_weights"
        });
    }
}