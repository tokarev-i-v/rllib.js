﻿      //is there a helper function that shows the local xyz?
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
          this.reward_bonus = 0.0;
          this.digestion_signal = 0.0;
          // outputs on world
          this.rot = 0.0; // rotation speed of 1st wheel
          this.speed = 0.0;
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
        
        onAdding(params){
          this.scene = params.scene;
          this.body = new BABYLON.MeshBuilder.CreateBox(
            "box",
            { height: 1, width: 1, depth: 1 },
            this.scene
          );
          this.body.material = new BABYLON.StandardMaterial(this.scene);
          this.body.material.diffuseColor = new BABYLON.Color3(1.0, 0, 0); 
          this.body.position.y = 0;
          this.body.isPickable = this.isPickable;
        }

        //one final test!
        moveNorth() {
          this.body.position.z = this.body.position.z + 0.1;
        }
        moveSouth() {
          this.body.position.z = this.body.position.z - 0.1;
        }
        moveEast() {
          this.body.position.x = this.body.position.x + 0.1;
        }
        moveWest() {
          this.body.position.x = this.body.position.x - 0.1;
        }
        rotateRight() {
          this.body.rotation.y = this.body.rotation.y + 0.01;
        }
        rotateLeft() {
          this.body.rotation.y = this.body.rotation.y - 0.01;
        }
        vecToLocal(vector, mesh) {
          var m = mesh.getWorldMatrix();
          var v = BABYLON.Vector3.TransformCoordinates(vector, m);
          return v;
        }

        // createScene.scene
        // divide it into smaller functions
        // one responsibility or not?
        castRay(arg) {
          
          
          //const or let
          //const if the variable will not be changed (const dies after the function is called)
          //let, so they can be changed
          // (var lives for a long period do not use as it causes memory leaks)
                    
          var origin = this.body.position;
          var length = 10;
          
          const forward = this.body.getDirection(BABYLON.Vector3.Forward());
          const left = this.body.getDirection(new BABYLON.Vector3(-1 * this.eye_view_radius, 0, 1));
          const right = this.body.getDirection(new BABYLON.Vector3(this.eye_view_radius, 0, 1));
          
          //here are the rays
          var ray = new BABYLON.Ray(origin, forward, length);
          var rayTwo = new BABYLON.Ray(origin, left, length); 
          var rayThree = new BABYLON.Ray(origin, right, length); 
          //var rayThree = new BABYLON.Ray(origin, direction, length);

          //here are the ray helpers & here we dispose them. 
          if(this.rayHelper) {
            this.rayHelper.dispose();
          }

          if(this.rayHelperTwo) {
            this.rayHelperTwo.dispose();
          }
          
          if(this.rayHelperThree) {
            this.rayHelperThree.dispose();
          }
          
          this.rayHelper = new BABYLON.RayHelper(ray);
          this.rayHelperTwo = new BABYLON.RayHelper(rayTwo);
          this.rayHelperThree = new BABYLON.RayHelper(rayThree);
          //somehow these variables reverse the mouse direction...

          //showing the rays with a ray helper
          // this.rayHelper.attachToMesh(
          //   this.body,            
          //   forward,
          //   this.body.localMeshOrigin,
          //   length
          // );
          this.rayHelper.show(arg);

          // this.rayHelperTwo.attachToMesh(
          //   this.body,
          //   left,
          //   this.body.localMeshOrigin,
          //   length
          // );
          this.rayHelperTwo.show(arg);
          
          // this.rayHelperThree.attachToMesh(
          //   this.body,
          //   right,
          //   this.body.localMeshOrigin,
          //   length
          // );
          this.rayHelperThree.show(arg);
          
          //hit detection and print out
          var hit = arg.pickWithRay(ray);
          if (hit.pickedMesh) {
            console.log(hit.pickedMesh.id);
          }
          var hitTwo = arg.pickWithRay(rayTwo);
          if (hitTwo.pickedMesh) {
            console.log(hitTwo.pickedMesh.id);
          }
          var hitTwo = arg.pickWithRay(rayThree);
          if (hitTwo.pickedMesh) {
            console.log(hitTwo.pickedMesh.id);
          }
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
          //agent
          this.agent;
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
                
                //agent casts a ray
                
                //something here is resetting the position;
                for (let agent of this.agents){
                  agent.castRay(this.sceneToRender);
                }
                
                
                //these are MOVEMENT TESTS
                //agent.moveNorth();
                //agent.rotateRight();
                
                
                this.sceneToRender.render();
              }
            }.bind(this));
          });


          // Resize
          window.addEventListener("resize", function() {
            this.engine.resize();
          }.bind(this));
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

        start() {
          requestAnimationFrame(this.start.bind(this));
          this.step();
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
  
          this.agents = [];

          //AGENT INITIALIZATION
          // this.agent = new Agent({"scene": this.scene});
          // this.agent.body.isPickable = false;
  
          //this is a helper to debug agent motion
          // this.mousemovef = function() {
          //   var pickResult = this.scene.pick(this.scene.pointerX, this.scene.pointerY);
          //   if (pickResult.hit) {
          //     var diffX = pickResult.pickedPoint.x - this.agent.body.position.x;
          //     var diffY = pickResult.pickedPoint.z - this.agent.body.position.z;
          //     this.agent.body.rotation.y = Math.atan2(diffX, diffY);
          //   }
          // }
          // this.scene.onPointerMove = function() {
          //   this.mousemovef();
          // }.bind(this);
          //making available to global scope for update
          //scene.ray = castRay;
  
          //////////
  
          //FRUITS AND POISONS INITIALIZATION
          for (var i = 0; i < 10; i++) {
            const flips = [-1, 1];
            function randomFlip(flips) {
              return flips[Math.floor(Math.random() * flips.length)];
            }
            var randomX = Math.random() * 50 * randomFlip(flips);
            var randomZ = Math.random() * 50 * randomFlip(flips);
            var apple = new Apple({"x": randomX, "z": randomZ, "scene": this.scene});
          }
          for (var i = 0; i < 10; i++) {
            const flips = [-1, 1];
            function randomFlip(flips) {
              return flips[Math.floor(Math.random() * flips.length)];
            }
            var randomX = Math.random() * 50 * randomFlip(flips);
            var randomZ = Math.random() * 50 * randomFlip(flips);
            var poison = new Poison({"x": randomX, "z": randomZ, "scene": this.scene});
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

