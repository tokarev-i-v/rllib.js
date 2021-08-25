﻿/**
 * @class Eye
 * It presents as agent's eye detector.
 */
 class Eye{
  /**
   * 
   * @param {THREE.Vector3} agent_pos_vec Vector that would use as src
   * vector for raycastring
   * @param {Number} alpha angle 
   * @param {Number} r radius
   */
  constructor(a, alpha, r){
 
    this.default_positions = [];
    const colors = [];

    this.default_positions.push(new BABYLON.Vector3(0, 0, 0 ));
    this.default_positions.push(new BABYLON.Vector3(Math.sin(Math.PI*alpha/180)*r,0,  Math.cos(Math.PI*alpha/180)*r));
    
    this.max_range = 20;
    this.sensed_proximity = 20;
    this.a = a;
    this._view = BABYLON.MeshBuilder.CreateLines("lines", {
      points: this.default_positions,
    });
    this.vec_to_add = a._view.getDirection(new BABYLON.Vector3(0,0,1)).subtract(this.default_positions[1]);

  }

  get view(){
    return this._view;
  }

  get_detection(params) {
    
    let origin = this.a._view.position;

    let end_vec = this.a._view.forward.add(this.vec_to_add);


    let scene = params.scene;
    //here are the rays
    let ray = new BABYLON.Ray(origin, end_vec, this.sensed_proximity);
    
    //hit detection and print out
    let hit = scene.pickWithRay(ray);
    // if (hit.pickedMesh) {
    //   console.log(hit.pickedMesh.id);
    // }

  }
}


//is there a helper function that shows the local xyz?
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      class Agent {
        constructor(opt) {
          this.isPickable = false;
          this.eye_view_radius = 2;

          //saving references allows us to dispose the ray helpers once we create new ones. Not disposing caused the multiple rays being attached to the mesh.
          this.rayHelper;
          this.rayHelperTwo;
          this.rayHelperThree;   

          this.eyes_count = opt.eyes_count;

          this.action_space = new BoxSpace(this.min_action,this.max_action, [3]);
          this.observation_space = new BoxSpace(-10000000, 100000000, [this.eyes_count * 3])
          this.eyes = [];

          let r = 20;
          let dalpha = 10;
          let alpha = -(dalpha*this.eyes_count)/2;          
          
          this.reward_bonus = 0.0;
          this.digestion_signal = 0.0;

          this.rot = 0.0; //rotation angle
          this.speed = 0.0; // movement speed

          this._view = new BABYLON.MeshBuilder.CreateBox(
            "box",
            { height: 1, width: 1, depth: 1 },
            // this.scene
          );
          this._view.material = new BABYLON.StandardMaterial();
          this._view.material.diffuseColor = new BABYLON.Color3(1.0, 0, 0); 
          this._view.position.y = 0;
          this._view.isPickable = this.isPickable;

          /**Now we create agent's eyes*/
          for (let i = 0; i < this.eyes_count; i++){
            let eye = new Eye(this, alpha, r);
            let mesh = eye.view;
            this.view.addChild(mesh);
            this.eyes.push(eye);
            alpha += dalpha;
          }
          this._frontEye = null;
          if(this.eyes.length % 2 === 0){
            this._frontEye = this.eyes[Math.round(this.eyes.length/2)];
          }else {
            this._frontEye = this.eyes[Math.round(this.eyes.length/2)-1];
          }
    
          this.reward_bonus = 0.0;
          this.digestion_signal = 0.0;
          // outputs on world
          this.rot = 0.0; // rotation speed of 1st wheel
          this.speed = 0.0;
          this.average_reward_window = new Buffer(10, 1000);
          this.displayHistoryData = [];
          this.surface = { name: 'Mean reward', tab: 'Charts' };
          setInterval(this.graphic_vis.bind(this), 1000);
        }

        get view(){
          return this._view;
        }
        
        set view(v){
          this._view = v;
        }

        get position(){
          return this._view.position;
        }
        
        set position(pos){
          this._view.position = pos;
        }

        get rotation(){
          return this._view.rotation;
        }

        set rotation(rot){
          this._view.rotation = rot;
        }
        get angle(){
          return this._view.rotation.z;
        }
        set angle(val){
          this._view.rotation.z = val;
        }
        get frontEye(){
          return this._frontEye;
        }
        /**
         * Updates graphics
         */
        graphic_vis(){
          if (this.displayHistoryData.length > 1100){
            this.displayHistoryData.splice(0,100);
          }
          this.displayHistoryData.push({"x": this.age, "y": this.average_reward_window.get_average()});
          let data = {values: this.displayHistoryData};
          tfvis.render.linechart(this.surface, data);
        }
        get_observation() {
          let num_eyes = this.eyes.length;
          let obs = new Array(num_eyes * 3);
          for(let i=0;i<num_eyes;i++) {
            let e = this.eyes[i];
            obs[i*3] = 1.0;
            obs[i*3+1] = 1.0;
            obs[i*3+2] = 1.0;
            if(e.sensed_type !== -1) {
              // sensed_type is 0 for wall, 1 for food and 2 for poison.
              // lets do a 1-of-k encoding into the input array
              obs[i*3 + e.sensed_type] = e.sensed_proximity/e.max_range; // normalize to [0,1]
            }
          }
          return obs;
        }
        
        get_reward() {
          // compute reward 
          let proximity_reward = 0.0;
          let num_eyes = this.eyes.length;
          for(let i=0;i<num_eyes;i++) {
            var e = this.eyes[i];
            // Here could be
            // proximity_reward += e.sensed_type === 0 ? e.sensed_proximity/e.max_range : 0.0;
            // proximity_reward += e.sensed_type === 1 ? 1 - e.sensed_proximity : 0.0;
            // proximity_reward += e.sensed_type === 2 ? -(1 - e.sensed_proximity) : 0.0;
          }
          // console.log("num_eyes: %s ", num_eyes);    
          proximity_reward = proximity_reward/num_eyes;
          
          // agents like to go straight forward
          let forward_reward = 0.0;
          if(this.actionix === 0 && proximity_reward > 0.75) forward_reward = 0.1 * proximity_reward;
          
          // agents like to eat good things
          let digestion_reward = this.digestion_signal;
          this.digestion_signal = 0.0;
          let reward = proximity_reward + forward_reward + digestion_reward;   
          this.average_reward_window.add(reward);
          return reward;
        }
      

        onAdding(params){
          this.scene = params.scene;
          this._view = new BABYLON.MeshBuilder.CreateBox(
            "box",
            { height: 1, width: 1, depth: 1 },
            this.scene
          );
          this._view.material = new BABYLON.StandardMaterial(this.scene);
          this._view.material.diffuseColor = new BABYLON.Color3(1.0, 0, 0); 
          this._view.position.y = 0;
          this._view.isPickable = this.isPickable;
        }

        vecToLocal(vector, mesh) {
          var m = mesh.getWorldMatrix();
          var v = BABYLON.Vector3.TransformCoordinates(vector, m);
          return v;
        }


      }

      class Apple {
        constructor(params) {
          this.body = BABYLON.MeshBuilder.CreateSphere("apple", {});
          this.body.position.x = params.x;
          this.body.position.z = params.z;
          this.appleMaterial = new BABYLON.StandardMaterial(
            "myMaterial",
            params.scene
          );
          this.reward = 10;
          this.appleMaterial.emissiveColor = new BABYLON.Color3(0, 0, 1);
          this.body.material = this.appleMaterial;
        }
        get position(){
          return this.body.position;
        }
        set position(pos){
          this.body.position = pos;
        }
      }

      class Poison {
        constructor(params) {
          this.body = BABYLON.MeshBuilder.CreateBox("poison", {});
          this.body.position.x = params.x;
          this.body.position.z = params.z;
          this.poisonMaterial = new BABYLON.StandardMaterial(
            "myMaterial",
            params.scene
          );
          this.reward = -70;
          this.poisonMaterial.emissiveColor = new BABYLON.Color3(1, 0, 0);
          this.body.material = this.poisonMaterial;
        }
        get position(){
          return this.body.position;
        }
        set position(pos){
          this.body.position = pos;
        }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      class BabylonjsVRExperimentWorld{

        constructor(opt){

          this.average_reward_window = new Buffer(10, 1000);
          this.displayHistoryData = [];
          this.surface = { name: 'Mean reward', tab: 'Charts' };
          setInterval(this.graphic_vis.bind(this), 1000);
          this.canvas = document.getElementById("renderCanvas");
          this.engine = null;
          this.scene = null;
          this.sceneToRender = null;
          this.xr;
          this.xrCamera;
          this.agents = [];
          try {
            this.engine = this.createDefaultEngine(this.canvas);
          } catch (e) {
            console.log(
              "the available createEngine function failed. Creating the default engine instead"
            );
            this.engine = this.createDefaultEngine();
          }
          if (!this.engine) throw "engine should not be null.";
          let scenePromise = this.createScene();
          scenePromise.then(returnedScene => {
            this.sceneToRender = returnedScene;

            this.engine.runRenderLoop(function() {
              if (this.sceneToRender) {
                this.sceneToRender.render();
              }
            }.bind(this));
          });


          // Resize
          window.addEventListener("resize", function() {
            this.engine.resize();
          }.bind(this));
        }

      // helper function to get closest colliding walls/items
      computeCollisions(eye, check_walls, check_items) {
        let minres = false;

        if(check_walls) {
            let res = eye.get_detection({scene: this.scene, walls: this.walls});
            if(res) {
              res.type = 0; // 0 is wall
              if(!minres) { minres=res; }
            }
        }
        // collide with items
        if(check_items) {
          let res = eye.get_detection({scene: this.sceneToRender, items: this.items});
          if(res) {
            if(!minres) { minres=res; }
          }
        }
        return minres;
      }

        /**
         * Updates graphics
         */
        graphic_vis(){
          if (this.displayHistoryData.length > 1100){
            this.displayHistoryData.splice(0,100);
          }
          this.displayHistoryData.push({"x": this.age, "y": this.average_reward_window.get_average()});
          let data = {values: this.displayHistoryData};
          tfvis.render.linechart(this.surface, data);
        }

        createDefaultEngine(canvas) {
          return new BABYLON.Engine(canvas, true, {
            preserveDrawingBuffer: true,
            stencil: true
          });
        };

        addAgent(agent){
          agent.onAdding({scene: this.scene});
          this.agents.push(agent);
          this.n_obs = agent.get_observation();
        };

        step(action) {
          let ret = this.tick(action);
          this.rew_episode += ret[1];
          this.len_episode += 1;
          return ret;
        }

        tick(action) {

          this.agents[0].rot = action[0];
          this.agents[0].speed = action[1];  
    
          // if (action[2] > 0.5){
          //   let bullet = this.agents[0].fire();
          //   this.addBullet(bullet);
          // }
          if (this.need_reset_env){
            this.reset();
            this.need_reset_env = 0;
          }
          let state, done, reward;
    
          // tick the environment
          this.clock++;
          
          for(var i=0,n=this.agents.length;i<n;i++) {
            var a = this.agents[i];
            for(var ei=0,ne=a.eyes.length;ei<ne;ei++) {
              var e = a.eyes[ei];
              var res = this.computeCollisions(e, true, true);
              if(res) {
                // eye collided with wall
                e.sensed_proximity = res.dist;
                e.sensed_type = res.type;
              } else {
                e.sensed_proximity = e.max_range;
                e.sensed_type = -1;
              }
            }
          }
          let states = [];
          // let the agents behave in the world based on their input
          for(var i=0,n=this.agents.length;i<n;i++) {
            states.push(this.agents[i].get_observation());
          }
          
          // apply outputs of agents on evironment
          for(let i=0,n=this.agents.length;i<n;i++) {
            let a = this.agents[i];
            let v = new BABYLON.Vector3(0, 0, 1);
            a._view.getDirection(v);
            v = v.normalize();
            v = v.scale(a.speed);
            a.position = a.position.add(v);
            a.rotation.y += a.rot;
            
            let res = this.computeCollisions(a.frontEye, true, false);
            if(res) {
              a.position = a.op;
            }
            
            // handle boundary conditions
            if(a.position.x< -this.W/2)a.position.x=-this.W/2;
            if(a.position.x>this.W/2)a.position.x=this.W/2;
            if(a.position.z< -this.H/2)a.position.z=-this.H/2;
            if(a.position.z>this.H/2)a.position.z=this.H/2;
          }
          
          for(var i=0,n=this.items.length;i<n;i++) {
            var it = this.items[i];
            it.age += 1;
            
            // see if some agent gets lunch
            for(var j=0,m=this.agents.length;j<m;j++) {
              var a = this.agents[j];
              var d = BABYLON.Vector3.Distance(a.position, it.position);
              if(d < it.rad + a.rad) {
                
                var rescheck = this.computeCollisions(a.frontEye, true, false);
                if(!rescheck) { 
                  if(it.type === 1) a.digestion_signal += it.reward;
                  if(it.type === 2) a.digestion_signal += it.reward;
                  this.removeItem(it);
                  i--;
                  n--;
                  break; // break out of loop, item was consumed
                }
              }
            }
            
            if(it.age > 5000 && this.clock % 100 === 0 && getRandomArbitrary(0,1)<0.1) {
              this.removeItem(it);
              i--;
              n--;
            }
          }
          if(this.items.length < this.items_count) {
            this.generateItem();
          }
          let rewards = [];
          for(var i=0,n=this.agents.length;i<n;i++) {
            rewards.push(this.agents[i].get_reward());
          }
          done = 0;
    
          state = states[0];
          reward = rewards[0];
          let info = {};
          
          // reward -= this.clock;
          let ret_data = [state, reward, done, info];
          if(this.clock % 1000 == 0){
            done = true;
            ret_data[2] = done;
            this.need_reset_env = 1;
          }
          return ret_data; 
        }
        


        reset(){
          this.n_obs = this.agents[0].get_observation();
          this.rew_episode = 0;
          this.len_episode = 0;
          this.clock = 0;
          return this.n_obs.slice()
        }
  
        get_episode_reward(){
          return this.rew_episode;
        }
    
        get_episode_length(){
          return this.len_episode;
        }

        async createScene() {
          // This creates a basic Babylon Scene object (non-mesh)
          this.scene = new BABYLON.Scene(this.engine);
          const gravityVector = new BABYLON.Vector3(0, -9.8, 0);
          const physicsPlugin = new BABYLON.CannonJSPlugin();
          this.scene.enablePhysics(gravityVector, physicsPlugin);
  
          this.camera = new BABYLON.FreeCamera(
            "camera1",
            new BABYLON.Vector3(0, 10, 80),
            this.scene
          );
          this.camera.setTarget(new BABYLON.Vector3(0, 10, 400));
          this.camera.attachControl(this.canvas, true);
  
          //light
          var light = new BABYLON.HemisphericLight(
            "light",
            new BABYLON.Vector3(0, 1, 0),
            this.scene
          );
          light.intensity = 0.7;
  
          //default environment
          var environment = this.scene.createDefaultEnvironment({
            createGround: false,
            skyboxSize: 10000000
          });
          environment.setMainColor(BABYLON.Color3.FromHexString("#74b9ff"));
  
          //default ground
          var ground = BABYLON.MeshBuilder.CreateGround(
            "ground",
            { width: 10000, height: 10000 },
            this.scene
          );
  
          //not exactly sure what this physics thing is
          ground.physicsImpostor = new BABYLON.PhysicsImpostor(
            ground,
            BABYLON.PhysicsImpostor.BoxImpostor,
            {
              mass: 0,
              friction: 0.8,
              restitution: 0.5,
              disableBidirectionalTransformation: true
            },
            this.scene
          );
          ground.checkCollisions = true;
  
          //messing with the ground color
          var groundMaterial = new BABYLON.GridMaterial("groundMaterial", this.scene);
          groundMaterial.majorUnitFrequency = 5;
          groundMaterial.minorUnitVisibility = 0.45;
          groundMaterial.gridRatio = 2;
          groundMaterial.backFaceCulling = false;
          groundMaterial.mainColor = new BABYLON.Color3(1, 1, 1);
          groundMaterial.lineColor = new BABYLON.Color3(1.0, 1.0, 1.0);
          groundMaterial.opacity = 0.98;
          ground.material = groundMaterial; //new BABYLON.GridMaterial("mat", scene);
    
          this.items = [];

          //FRUITS AND POISONS INITIALIZATION
          for (var i = 0; i < 10; i++) {
            const flips = [-1, 1];
            function randomFlip(flips) {
              return flips[Math.floor(Math.random() * flips.length)];
            }
            var randomX = Math.random() * 50 * randomFlip(flips);
            var randomZ = Math.random() * 50 * randomFlip(flips);
            var apple = new Apple({"x": randomX, "z": randomZ, "scene": this.scene});
            this.items.push(apple);
          }
          for (var i = 0; i < 10; i++) {
            const flips = [-1, 1];
            function randomFlip(flips) {
              return flips[Math.floor(Math.random() * flips.length)];
            }
            var randomX = Math.random() * 50 * randomFlip(flips);
            var randomZ = Math.random() * 50 * randomFlip(flips);
            var poison = new Poison({"x": randomX, "z": randomZ, "scene": this.scene});
            this.items.push(poison);
          }
  
          // enable xr
          this.xr = await this.scene.createDefaultXRExperienceAsync({
            floorMeshes: [ground]
            //multiView: true
            //useMultiview:true
          });
  
          // enable physics
          this.xrPhysics = this.xr.baseExperience.featuresManager.enableFeature(
            BABYLON.WebXRFeatureName.PHYSICS_CONTROLLERS,
            "latest",
            {
              xrInput: this.xr.input,
              physicsProperties: {
                restitution: 0.5,
                impostorSize: 0.1,
                impostorType: BABYLON.PhysicsImpostor.BoxImpostor
              },
              enableHeadsetImpostor: true
            }
          );
  
          //XR-way of interacting with the controllers for the left hand:
          this.xr.input.onControllerAddedObservable.add(controller => {
            controller.onMotionControllerInitObservable.add(motionController => {
              if (motionController.handness === "left") {
                motionController
                  .getMainComponent()
                  .onButtonStateChangedObservable.add(component => {
                    if (component.changes.pressed) {
                      if (component.pressed) {
                        //console.log("hi");
                        //POV = false;
                      }
                    }
                  });
              }
            });
          });
          this.xr.input.onControllerAddedObservable.add(controller => {
            controller.onMotionControllerInitObservable.add(motionController => {
              if (motionController.handness === "right") {
                motionController
                  .getMainComponent()
                  .onButtonStateChangedObservable.add(component => {
                    if (component.changes.pressed) {
                      if (component.pressed) {
                        //console.log("hi");
                        //POV = true;
                      }
                    }
                  });
              }
            });
          });
          return this.scene;  
        };
      
      }

      //something got fucked up with the canvas re-sizing
      //canvas.height = 1000;
      //canvas.width = 1000;




      /**
       * Physics WebXR playground.
       * Objects can be picked using the squeeze button of the left controller (if available) and can be thrown.
       * The left trigger resets the scene, right trigger shoots a bullet straight.
       *
       * Both hands and the headset have impostors, so you can touch the objects, move them, headbutt them.
       *
       * Use the boxing area in the center to understand how the hand and head impostors work.
       *
       * Created by Raanan Weber (@RaananW)
       */

