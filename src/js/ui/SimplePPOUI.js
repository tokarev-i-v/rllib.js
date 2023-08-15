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
            this.setWeightsButton.style.top = "300px";
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
            this.policyWeightsJSON = document.createElement("input");
            this.policyWeightsJSON.type = "file";
            this.policyWeightsJSON.style.left = "0px";
            this.policyWeightsJSON.style.top = "100px";          
            this.policyWeightsJSON.id = "ui_policy_weights_json_input";
            this.policyWeightsJSON.style.zIndex = 10;
            this.policyWeightsJSON.style.position = 'absolute';


            this.policyWeights = document.createElement("input");
            this.policyWeights.type = "file";
            this.policyWeights.style.left = "0px";
            this.policyWeights.style.top = "150px";          
            this.policyWeights.id = "ui_policy_weights_input";
            this.policyWeights.style.zIndex = 10;
            this.policyWeights.style.position = 'absolute';

            if(this.parent){
                this.parent.appendChild(this.policyWeightsJSON);
                this.parent.appendChild(this.policyWeights);
            } else {
                document.body.appendChild(this.policyWeightsJSON);
                document.body.appendChild(this.policyWeights);
                this.parent = document.body;
            }
        }
    }

    initValueWeightsPathInput(){
        if(document){
            this.valueWeightsJSON = document.createElement("input");
            this.valueWeightsJSON.type = "file";
            this.valueWeightsJSON.style.left = "0px";
            this.valueWeightsJSON.style.top = "200px";          
            this.valueWeightsJSON.id = "ui_value_weights_json_input";
            this.valueWeightsJSON.style.zIndex = 10;
            this.valueWeightsJSON.style.position = 'absolute';


            this.valueWeights = document.createElement("input");
            this.valueWeights.type = "file";
            this.valueWeights.style.left = "0px";
            this.valueWeights.style.top = "250px";          
            this.valueWeights.id = "ui_value_weights_input";
            this.valueWeights.style.zIndex = 10;
            this.valueWeights.style.position = 'absolute';

            if(this.parent){
                this.parent.appendChild(this.valueWeightsJSON);
                this.parent.appendChild(this.valueWeights);
            } else {
                document.body.appendChild(this.valueWeightsJSON);
                document.body.appendChild(this.valueWeights);
                this.parent = document.body;
            }
        }
    }


    async setWeights(){

        const uploadPolicyJSONInput = document.getElementById('ui_policy_weights_json_input');
        const uploadPolicyWeightsInput = document.getElementById('ui_policy_weights_input');
        let policy_model = await tf.loadLayersModel(tf.io.browserFiles(
         [uploadPolicyJSONInput.files[0], uploadPolicyWeightsInput.files[0]]));

        const uploadValueJSONInput = document.getElementById('ui_value_weights_json_input');
        const uploadValueWeightsInput = document.getElementById('ui_value_weights_input');
        let value_model = await tf.loadLayersModel(tf.io.browserFiles(
        [uploadValueJSONInput.files[0], uploadValueWeightsInput.files[0]]));
 
        this.worker.postMessage({
            msg_type: "set_weigths",
            policy_weights: get_serialized_layers_data(policy_model),
            value_weights: get_serialized_layers_data(value_model),
        });
    }

    downloadWeights(){
        this.worker.postMessage({
            msg_type: "get_policy_weights"
        });
    }
}