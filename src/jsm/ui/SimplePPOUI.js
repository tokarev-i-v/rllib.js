import {params_setter} from '../utils';

export class SimpleUI{
    constructor(opt){
        this.params_setter = params_setter.bind(this);
        let default_options = {
            "nn_config": [128, 128],
            "policy_nn": null,
            "parent": null,
            "button_text": "Download agent weights"
        }
        this.params_setter(default_options, opt);
        
    }
    
    initDownloadButton(){
        if(document){
            this.downloadWeightsButton = document.createElement("Button");
            this.downloadWeightsButton.addEventListener("click", this.downloadWeights.bind(this));
            this.downloadWeightsButton.style.left = "0px";
            this.downloadWeightsButton.style.top = "50px";
            this.downloadWeightsButton.style.width = "100px";
            this.downloadWeightsButton.style.height = "40px";            
            this.downloadWeightsButton.id = "ppo_ui";
            this.downloadWeightsButton.style.zIndex = 10;
            this.downloadWeightsButton.style.position = 'absolute';
            this.downloadWeightsButton.textContent = this.button_text;
            
            if(this.parent){
                this.parent.appendChild(this.downloadWeightsButton);
            } else {
                document.body.appendChild(this.downloadWeightsButton);
                this.parent = document.body;
            }
        }
    }

    initLoadButton(){
        if(document){
            this.setWeightsButton = document.createElement("Button");
            this.setWeightsButton.addEventListener("click", this.setWeightsButton.bind(this));
            this.setWeightsButton.style.left = "0px";
            this.setWeightsButton.style.top = "50px";
            this.setWeightsButton.style.width = "100px";
            this.setWeightsButton.style.height = "40px";            
            this.setWeightsButton.id = "ppo_ui";
            this.setWeightsButton.style.zIndex = 10;
            this.setWeightsButton.style.position = 'absolute';
            this.setWeightsButton.textContent = this.button_text;



            
            if(this.parent){
                this.parent.appendChild(this.downloadButton);
            } else {
                document.body.appendChild(this.downloadButton);
                this.parent = document.body;
            }
        }
    }

    downloadWeights(){
        const saveResult = this.policy_nn.save('downloads://mymodel');
    }
}