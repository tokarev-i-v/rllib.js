/**
 * HungryWorld env.
 * 
 * Environment characteristics:
 *  action space: continuous;
 *  observation space: continuous;
 */

let CONSTANTS = {
  "TYPES": {
    "BULLET": 0,
    "POISON": 1,
    "APPLE": 2
  }
};


/**
 * Class describes targets, that must be eaten by agent.
 * :
 *  --reward: 0.99
 */
class Food {
  /**
   * 
   * @param {THREE.Vector3} pos position of the food 
   */
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0x11FF11})
    )
    this._view.position.x = 10;
    this.age = 0;
    this.type = 1;
    this.reward = 1;
    this.cleanup_ = false;
    this._view._rl = {
      type: this.type,
      obj: this
    }
    this._view.position.copy(pos);
  }
  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }
}

/**
 * Class describes targets, that must be eaten by agent.
 * Charasteristics:
 *  --reward: -0.99
 */
class Poison {
  /**
   * 
 * @param {THREE.Vector3} pos
   */
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0xFFF422})
    )      
    this.age = 0;
    this.type = 2;
    this.reward = -3;
    this.cleanup_ = false;
    this._view.position.copy(pos);
    this._view._rl = {
      type: this.type,
      obj: this
    }
  }
  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }
}


/**
 * Class describes bullet, that could be shotted by Agent.
 */
class Bullet {
  /**
   * 
   * @param {THREE.Vector3} pos
   * @param {THREE.Vector3} dir
   */
  constructor(pos, dir){
    this.speed = 15;
    this.dir = dir;
    this.rad = 0.2;
    this.way = new THREE.Vector3();
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0xFF0000})
    )      
    this._view.geometry.computeBoundingBox();
    this._view.position.copy(pos);
  }
  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }

  update(time){
    let v = this.dir.clone();
    
    if (time){
      v.multiplyScalar(this.speed*time);
    }else {
      v.multiplyScalar(this.speed);
    }
    this.position.add(v);
    this.way.add(v);
  }
}

/**
 * Agent from this world;
 */
class HungryAgent{
  /**
   * 
   * @param {Object} opt 
   */
  constructor(opt){
    this.rad = 2;
    this.imgshape = opt.imgshape;
    let materials = [
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // right
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // left
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // top
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // bottom
	    new THREE.MeshStandardMaterial( { map: THREE.ImageUtils.loadTexture('src/images/hunter_black_278.png') } ), // back
	    new THREE.MeshStandardMaterial( { color: 0x000000 } )  // front
  	];
    
    this.Container = document.createElement('div');

    this.Renderer = new THREE.WebGLRenderer();
    this.Renderer.setSize(this.imgshape[0], this.imgshape[1]);
    this.Container.appendChild(this.Renderer.domElement);
    this.Container.setAttribute("style", "position: absolute; z-index:100; left: 200px;");
    
    document.body.appendChild(this.Container);

    this.Scene = opt.Scene;
    this.Camera = new THREE.PerspectiveCamera(45, this.imgshape[0] / this.imgshape[1], 1, 10);    
    this._view = new THREE.Mesh(
      new THREE.BoxBufferGeometry(this.rad,this.rad,this.rad),
      new THREE.MultiMaterial( materials )
    );
    this.helper = new THREE.CameraHelper( this.Camera );
    this._view.add(this.Camera);
    this.Scene.add(this.helper);
    this.Camera.rotation.y -= Math.PI;

    this.min_action = -1.0;
    this.max_action = 1.0;

    this.hungry = 0;
    this.greens_count = 0;
    this.yellows_count = 0;
    this.position.y = 1;
    this.action_space = new BoxSpace(this.min_action,this.max_action, [3]);
    this.observation_space = new BoxSpace(-1, 1, imgshape[0]*this.imgshape[1]*this.imgshape[2]);
    this.eyes_count = 1;
    this.eyes = [];
    let r = 20;
    let dalpha = 10;
    let alpha = -(dalpha*this.eyes_count)/2;
    /**Now we create agent's eyes*/
    for (let i = 0; i < this.eyes_count; i++){
      let eye = new Eye(this, alpha, r);
      let mesh = eye.view;
      this.view.add(mesh);
      this.view.add(eye.sphere_point);
      this.eyes.push(eye);
      alpha += dalpha;
    }
    this._frontEye = null;
    if(this.eyes.length % 2 === 0){
      this._frontEye = this.eyes[Math.round(this.eyes.length/2)];
    }else {
      this._frontEye = this.eyes[Math.round(this.eyes.length/2)-1];
    }

    console.log("Observation space shape: ", this.observation_space.shape);
    if (opt.algo){
      this.brain = new opt.algo({imgshape: this.imgshape, num_actions: this.action_space.length});
    }
    
    this.reward_bonus = 0.0;
    this.digestion_signal = 0.0;
    // outputs on world
    this.rot = 0.0; // rotation speed of 1st wheel
    this.speed = 0.0;
    this.average_reward_window = new Buffer(10, 1000);
    this.displayHistoryData = [];
    this.displayHistoryGreensData = [];
    this.displayHistoryYellowsData = [];
    this.chartsGreensYellowsNames = ['Greens count', 'Yellows count'];
    this.chartsGYCNames = ['Greens/Yellows coeff'];
    this.chartsMRNames = ['Mean Reward'];
    this.displayHistoryEatenCoefficientData = [];
    this.surface = { name: 'Mean reward', tab: 'Charts' };
    this.greens_surface = { name: 'Greens count', tab: 'Charts' };
    this.yellows_surface = { name: 'Yellows count', tab: 'Charts' };
    this.eaten_coefficient_surface = { name: 'Eaten Green/Yellow coefficient', tab: 'Charts' };
    setInterval(this.graphic_vis.bind(this), 1000);
  }
  
  fire(){
    let dir = new THREE.Vector3(); 
    this._view.getWorldDirection(dir);
    let bullet = new Bullet(this.position.clone(), dir);
    return bullet;
  }

  /**
   * @returns {BoxSpace} sampled action
   */
  sample_actions(){
    return this.action_space.sample();
  }

  /**
   * Updates graphics
   */
  graphic_vis(){
    if (this.displayHistoryData.length > 200){
      this.displayHistoryData.splice(0,100);
      this.displayHistoryYellowsData.splice(0,100);
      this.displayHistoryGreensData.splice(0,100);
      this.displayHistoryEatenCoefficientData.splice(0, 100);
    }
    this.displayHistoryData.push({"x": this.age, "y": this.average_reward_window.get_average()});
    let data = {values: this.displayHistoryData, series: this.chartsMRNames};
    tfvis.render.linechart(this.surface, data);
    this.displayHistoryGreensData.push({"x": this.age, "y": this.greens_count});
    this.displayHistoryYellowsData.push({"x": this.age, "y": this.yellows_count});
    data = {values: [this.displayHistoryGreensData, this.displayHistoryYellowsData], series: this.chartsGreensYellowsNames};
    tfvis.render.linechart(this.greens_surface, data);
    this.displayHistoryEatenCoefficientData.push({"x": this.age, "y": this.greens_count/(this.yellows_count+1)});
    data = {values: [this.displayHistoryEatenCoefficientData], series: this.chartsGYCNames};
    tfvis.render.linechart(this.eaten_coefficient_surface, data);
  }
  /**
   * 
   */
  get_observation() {
    this.Renderer.render(this.Scene, this.Camera);
    let gl = this.Renderer.getContext("webgl", {preserveDrawingBuffer: true});
    let pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let rearr = [];
    if (this.imgshape[2] === 3){
      for (let i = 0; i < gl.drawingBufferWidth; i++){
        rearr.push([]);
        for (let j = 0; j < gl.drawingBufferHeight; j++){
          rearr[i].push([pixels[i*4*gl.drawingBufferWidth+j*4+0]/255.0, pixels[i*4*gl.drawingBufferWidth+j*4+1]/255.0, pixels[i*4*gl.drawingBufferWidth+j*4+2]/255.0]);
        }
      }
    } else{
      for (let i = 0; i < gl.drawingBufferWidth; i++){
        rearr.push([]);
        for (let j = 0; j < gl.drawingBufferHeight; j++){
          rearr[i].push([(pixels[i*4*gl.drawingBufferWidth+j*4+0]/255.0 + pixels[i*4*gl.drawingBufferWidth+j*4+1]/255.0 + pixels[i*4*gl.drawingBufferWidth+j*4+2]/255.0)/3 ]);
        }
      }      
    }
    return rearr;
  }

  get_reward() {
    // compute reward 
    let proximity_reward = 0.0;
    
    // agents like to go straight forward
    let forward_reward = 0.0;
    if(this.actionix === 0 && proximity_reward > 0.75) forward_reward = 0.1 * proximity_reward;
    
    // agents like to eat good things
    let digestion_reward = this.digestion_signal;
    this.digestion_signal = 0.0;
    let reward = proximity_reward + forward_reward + digestion_reward;   
    if (reward > 0){
      this.hungry = 0;
    } else {
      this.hungry = -0.01;
    }
    reward += this.hungry;
    this.average_reward_window.add(reward);
    return reward;
  }

  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }

  get rotation(){
    return this._view.rotation;
  }
  set rotation(vec){
    this._view.rotation.copy(vec);
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
}
/**
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

    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial( { color: 0xffffff, linewidth: 3 } );    
    const positions = [];
    const colors = [];

    positions.push( 0, 0, 0 );
    positions.push(Math.sin(Math.PI*alpha/180)*r,0,  Math.cos(Math.PI*alpha/180)*r)

    
    geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
		// geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( colors, 3 ) );
    this._view = new THREE.Line(
      geometry,
      material
    );
    this.sphere_point = new THREE.Mesh(
      new THREE.SphereBufferGeometry(0.1, 10, 10),
      new THREE.MeshBasicMaterial({color:0x000000})
    )
    this.sphere_point.geometry.computeBoundingBox();
    this.sphere_point.position.set(Math.sin(Math.PI*alpha/180)*r, 0, Math.cos(Math.PI*alpha/180)*r);
    this.end_position = new THREE.Vector3(Math.sin(Math.PI*alpha/180)*r,Math.cos(Math.PI*alpha/180)*r,  Math.PI/2);
    /**setting Eye position and rotation */
    this._view.geometry.computeBoundingBox();
    this.raycaster = new THREE.Raycaster();
    this.max_range = 20;
    this.sensed_proximity = 20; // what the eye is seeing. will be set in world.tick()
    this.a = a;
  }
  get view(){
    return this._view;
  }
  /**
   * This function return the nearest detected object.
   * @param {THREE.Mesh[]} targets array of intersection targets
   * @returns {Object|null} 
   */
  get_detection(targets_objs){
    let targets = targets_objs.map((el)=>{
        return el.view;
    });
    let dst = new THREE.Vector3();
    dst.setFromMatrixPosition( this.sphere_point.matrixWorld );
    dst.sub(this.a.position.clone());
    dst.normalize();
    this.raycaster.set(this.a.position, dst);
    let intersects = this.raycaster.intersectObjects(targets);
    if (intersects.length > 0 && intersects[0].distance < this.max_range){
      // intersects[0].object.material.color.setHex( 0x0000ff );
      return {obj: intersects[0].object, type: intersects[0].object._rl.type, dist: intersects[0].distance}
    } else {
      return null;
    }
  }
}

  /**
   * @class
   * World Contains all features.
   */
class HungryWorld2D {

    /**
     * 
     * @param {*} opt parameters
     * @param opt.algorithm RL algorithm
     * @param opt.width worlds width
     * @param opt.height worlds height
     * @param opt.items_count items count in the world
     * 
     */
    constructor(opt){
      this.params_setter = params_setter.bind(this);

      this._default_params = {
        "items_count": 500,
        "W": 80,
        "H": 80,
        "algorithm": null,
        "UI": null,
        "imgshape": [64, 64, 3]
      };

      this.params_setter(this._default_params, opt);

      this.init();
      this.agents = [];
      this.clock = 0;
      this.walls = []; 

      this.rew_episode = 0;
      this.len_episode = 0;
      this.need_reset_env = 0;
      this.raycaster = new THREE.Raycaster();
      this.bullets = [];

      // set up food and poison
      this.items = []

      for(let k=0;k<this.items_count;k++) {
        this.generateItem();
      }

      if(this.algorithm){
        let agent = new Agent({eyes_count: 10, algo: this.algorithm});
        this.Scene.add(agent.view);
        this.agents.push(agent);        
      }
      this.render();
    }   
    /**
     * 
     * @param {*} agent RL agent algorithm
     */
    addAgent(agent){
      this.Scene.add(agent.view);
      this.agents.push(agent);
      this.n_obs = this.agents[0].get_observation();
    }
    /**
     * generates food and poison
     */
    generateItem(){
      let x = getRandomArbitrary(-this.W, this.W);
      let y = getRandomArbitrary(-this.H, this.H);
      let t = getRandomInt(1, 3); // food or poison (1 and 2)
      let it = null;
      if (t == 1){
        it = new Food(new THREE.Vector3(x, 1, y));
      }
      else{
        it = new Poison(new THREE.Vector3(x, 1, y));
      }
      this.items.push(it);
      this.Scene.add(it.view);
    }

  init(json_params){
    this.Container = document.createElement("div");
    this.Container.id = "MainContainer";
    this.Container.classList.add("Container");
    
    this.Renderer = new THREE.WebGLRenderer();
    this.Renderer.setSize(window.innerWidth, window.innerHeight);
    this.Container.appendChild(this.Renderer.domElement);

    document.body.insertBefore( this.Container, document.body.firstChild);

    this.stats = new Stats();
    document.body.appendChild(this.stats.dom);

    this.Camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 10000);



    window.addEventListener("resize", this.onWindowResize.bind(this), false);

    this.Camera.position.set(0,10, 10);
    this.Scene = new THREE.Scene();
    this.Scene.background = new THREE.Color( 0xaaccff );
    this.Scene.fog = new THREE.FogExp2( 0xaaccff, 0.007 );

    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    if (check){
      // this.CameraObj = new THREE.Object3D();
      // this.CameraObj.add(this.Camera);
      // this.Controls = new MobileControls({Camera: this.Camera, Object3D: this.CameraObj, Hammer: Hammer});      
      // this.Scene.add(this.CameraObj);

      this.Controls = new THREE.MapControls( this.Camera, document.getElementById("MainContainer") );

      //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

      this.Controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
      this.Controls.dampingFactor = 0.05;

      this.Controls.screenSpacePanning = false;

      this.Controls.minDistance = 10;
      this.Controls.maxDistance = 500;

      this.Controls.maxPolarAngle = Math.PI / 2;

    }else {

      this.Controls = new THREE.MapControls( this.Camera, document.getElementById("MainContainer") );

      //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

      this.Controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
      this.Controls.dampingFactor = 0.05;

      this.Controls.screenSpacePanning = false;

      this.Controls.minDistance = 10;
      this.Controls.maxDistance = 500;

      this.Controls.maxPolarAngle = Math.PI / 2;
    }

    this.Loader = new THREE.ColladaLoader();

    this.AmbientLight = new THREE.AmbientLight(0xFFFFFF, 0.9);
    this.Scene.add(this.AmbientLight);

    
    this.Clock = new THREE.Clock();

    if(typeof(document) !== typeof(undefined)){
      let TextureLoader = new THREE.TextureLoader();
      TextureLoader.load("src/images/grass.png", function (tex) {
          tex.wrapS = THREE.RepeatWrapping;
          tex.wrapT = THREE.RepeatWrapping;
          tex.repeat.set(100, 100);
          let ground = new THREE.Mesh(new THREE.PlaneBufferGeometry(1000, 1000), new THREE.MeshBasicMaterial({map: tex, side:THREE.DoubleSide}));
          ground.rotation.x -= Math.PI/2;
          this.Scene.add(ground);
      }.bind(this));

    }
  }
    onWindowResize() {
      this.Camera.aspect = window.innerWidth / window.innerHeight;
      this.Camera.updateProjectionMatrix();
      this.Renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /*Функция проверяет пересечение пуль с уничтожаемыми объектами*/
    controlBulletCollision (bullet, targets_objs)
    {
      let targets = targets_objs.map((el)=>{
          return el.view;
      });
      let dst = new THREE.Vector3();
      dst.setFromMatrixPosition( bullet.view.matrixWorld );
      dst.add(bullet.position.clone().negate());
      dst.normalize();
      this.raycaster.set(bullet._view.position, dst);
      let intersects = this.raycaster.intersectObjects(targets);
      if (intersects.length > 0 && intersects[0].distance < 20){
        return intersects[0].object._rl.obj;
      } else {
        return null;
      }
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


    render () {
      requestAnimationFrame(this.render.bind(this));
      this.stats.update();
      
      this.Renderer.render(this.Scene, this.Camera);
      let delta = this.Clock.getDelta();
      
      this.Controls.update(delta);
      for (let el of this.bullets){
        el.update(delta);
        let it = this.controlBulletCollision(el, this.items);
        if(it){
          if(it.type === 1) this.agents[0].digestion_signal += it.reward;
          if(it.type === 2) this.agents[0].digestion_signal += it.reward;
          this.removeItem(it);
          this.removeBullet(el);
        } else if (el.way.length() > 20){
          this.removeBullet(el);
          this.agents[0].digestion_signal += -10;
        }
      }
    }

    // helper function to get closest colliding walls/items
    computeCollisions(eye, check_walls, check_items) {
      let minres = false;

      if(check_walls) {
          let res = eye.get_detection(this.walls);
          if(res) {
            res.type = 0; // 0 is wall
            if(!minres) { minres=res; }
          }
      }
      // collide with items
      if(check_items) {
        let res = eye.get_detection(this.items);
        if(res) {
          if(!minres) { minres=res; }
        }
      }
      return minres;
    }

    removeItem(it){
      this.Scene.remove(it.view);
      this.items.splice(this.items.indexOf(it), 1);
    }

    step(action) {
      let ret = this.tick(action);
      this.rew_episode += ret[1];
      this.len_episode += 1;
      return ret;
    }

    addBullet(bullet){
      this.bullets.push(bullet);
      this.Scene.add(bullet.view);
    }
    removeBullet(bullet){
      let index = this.bullets.indexOf(bullet);
      this.bullets.splice(index, 1);
      this.Scene.remove(bullet.view);
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
      let states = [];
      // let the agents behave in the world based on their input
      for(let i=0,n=this.agents.length;i<n;i++) {
        states.push(this.agents[i].get_observation());
      }
      
      // apply outputs of agents on evironment
      for(let i=0,n=this.agents.length;i<n;i++) {
        let a = this.agents[i];
        let v = new THREE.Vector3();
        a._view.getWorldDirection(v);
        v.normalize();
        v.multiplyScalar(a.speed);
        a.position.add(v);
        a.rotation.y += a.rot;
        
        // handle boundary conditions
        if(a.position.x< -this.W/2)a.position.x=-this.W/2;
        if(a.position.x>this.W/2)a.position.x=this.W/2;
        if(a.position.z< -this.H/2)a.position.z=-this.H/2;
        if(a.position.z>this.H/2)a.position.z=this.H/2;
      }
      
      for(let i=0,n=this.items.length;i<n;i++) {
        let it = this.items[i];
        it.age += 1;
        
        // see if some agent gets lunch
        for(let j=0,m=this.agents.length;j<m;j++) {
          let a = this.agents[j];
          let d = a.position.distanceTo(it.position);
          if(d < it.rad + a.rad) {
            
            let rescheck = this.computeCollisions(a.frontEye, true, false);
            if(!rescheck) { 
              if(it.type === 1) {
                a.digestion_signal += it.reward;
                a.greens_count += 1;
              }
              if(it.type === 2) {
                a.digestion_signal += it.reward;
                a.yellows_count += 1;
              }
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
      for(let i=0,n=this.agents.length;i<n;i++) {
        rewards.push(this.agents[i].get_reward());
      }
      done = 0;

      state = states[0];
      reward = rewards[0];
      let info = {};
      
      let ret_data = [state, reward, done, info];
      if(this.clock % 1000 == 0){
        done = true;
        ret_data[2] = done;
        this.need_reset_env = 1;
      }
      return ret_data; 
    }
  }
  
