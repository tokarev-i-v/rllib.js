/*
On Desktop computer version we use THREE.FlyControls.
On Mobile phones we using Our Controls System.

IN: 

json_params = {
	Camera: THREE.Camera,
	Object3D: THREE.Object3D
};
*/
import * as THREE from '../../src/jsm/threejs/three.module';
import {DeviceOrientationControls} from "./threejs/DeviceOrientationControls";
import * as Hammer from 'hammerjs';
window.Hammer = Hammer.default;

export class MobileControls{
	constructor(json_params){
		if(json_params)
		{
			this.Camera = json_params.Camera;
			this.Object3D = json_params.Object3D;
		}
		this.touchUpdate = this.touchUpdate.bind(this);
		this.testDeviceMotion = this.testDeviceMotion.bind(this);

		this.AntiVec = new THREE.Vector3();

		this.OrientationParameters = {
			alpha: 0,
			beta: 0,
			gamma: 0,
			deviceMotionInterval: 0,
			
			touchRotRadX: 0,
			touchRotRadY: 0,
			touchDeltaTime: 0,
			
			shakeTimer: 0,
			shakeTimerStep: 0.1
		};


		this.AccelerometerControlButton = document.createElement("button");
		this.AccelerometerControlButton.setAttribute("id", "AccelerometerControlButton");
		this.AccelerometerControlButton.className = "AccelerometerControlButton";
		document.body.appendChild(this.AccelerometerControlButton);

		this.TouchControlButton = document.createElement("button");
		this.TouchControlButton.setAttribute("id", "TouchControlButton");
		this.TouchControlButton.className = "TouchControlButton";
		document.body.appendChild(this.TouchControlButton);

		this.AccelerometerControlButton.addEventListener("click", function () {
			this.TouchControlButton.style.visibility = "visible";
			this.AccelerometerControlButton.style.visibility = "hidden";
			this.RotateHammer.enable = false;
			this.update = this.AccelerometerControls.update;
		}.bind(this));

		this.TouchControlButton.addEventListener("click", function () {
			this.TouchControlButton.style.visibility = "hidden";
			this.AccelerometerControlButton.style.visibility = "visible";
			this.RotateHammer.enable = true;
			this.update = this.touchUpdate;
		}.bind(this));

		this.FrontMovingButton = document.createElement("div");
		this.FrontMovingButton.setAttribute("id", "FrontMovingButton");
		this.FrontMovingButton.className = "FrontMovingButton";
		document.body.appendChild(this.FrontMovingButton);

		this.FrontMovingButton.onmousedown = function (event)
		{
			this.FrontMovingOn = true;
		}.bind(this);

		this.FrontMovingButton.ontouchstart = function (event)
		{
			this.FrontMovingOn = true;
		}.bind(this);

		this.FrontMovingButton.onmouseup = function (event)
		{
			this.FrontMovingOn = false;
		}.bind(this);

		this.FrontMovingButton.ontouchend = function (event)
		{
			this.FrontMovingOn = false;
		}.bind(this);

		this.BackMovingButton = document.createElement("div");
		this.BackMovingButton.setAttribute("id", "BackMovingButton");
		this.BackMovingButton.className = "BackMovingButton";
		document.body.appendChild(this.BackMovingButton);
		this.BackMovingButton.onmousedown = function (event)
		{
			this.BackMovingOn = true;
		}.bind(this);

		this.BackMovingButton.ontouchstart = function (event)
		{
			this.BackMovingOn = true;
		}.bind(this);

		this.BackMovingButton.onmouseup = function (event)
		{
			this.BackMovingOn = false;
		}.bind(this);

		this.BackMovingButton.ontouchend = function (event)
		{
			this.BackMovingOn = false;
		}.bind(this);



		//mouse/touch controller
		this.RotateHammer = new window.Hammer(document.body);
		this.RotateHammer.get('pan').set({ direction: Hammer.DIRECTION_ALL });
		this.RotateHammer.on("pan", function (event) {

			this.OrientationParameters.touchRotRadX = THREE.Math.degToRad(event.deltaX/30);
			this.OrientationParameters.touchRotRadY = THREE.Math.degToRad(event.deltaY/30);
			this.OrientationParameters.touchDeltaTime = event.deltaTime;

		}.bind(this));


		this.AccelerometerControls = new DeviceOrientationControls(this.Camera);
		this.update = this.accelerometerUpdate;


		///////////////////////////////////////////////
		// Определяем, поддерживает ли устройство акселерометр!
		if (window.DeviceMotionEvent != undefined) {
			window.addEventListener("devicemotion", this.testDeviceMotion);
		} else {
			document.body.removeChild(this.AccelerometerControlButton);
			document.body.removeChild(this.TouchControlButton);
		}
	}
	accelerometerUpdate(delta)
	{
		this.AccelerometerControls.update();
	}

	touchUpdate(delta){

		this.Object3D.rotation.y -= this.OrientationParameters.touchRotRadX*delta;
		this.Object3D.rotation.x -= this.OrientationParameters.touchRotRadY*delta;	


		if(Math.abs(this.OrientationParameters.touchRotRadX) > 0.005)
		{
			this.OrientationParameters.touchRotRadX -= this.OrientationParameters.touchRotRadX/10;			
		} else if(this.OrientationParameters.touchRotRadX)
			this.OrientationParameters.touchRotRadX = 0;

		if(Math.abs(this.OrientationParameters.touchRotRadY) > 0.005)
		{
			this.OrientationParameters.touchRotRadY -= this.OrientationParameters.touchRotRadY/10;			
		} else if(this.OrientationParameters.touchRotRadY)
			this.OrientationParameters.touchRotRadY = 0;	

		if(this.FrontMovingOn)
		{
			this.Camera.getWorldDirection(this.AntiVec);
			this.AntiVec.normalize();
			this.AntiVec.multiplyScalar(5);
			this.Object3D.position.add(this.AntiVec);				
		}

		if(this.BackMovingOn)
		{
			this.Camera.getWorldDirection(this.AntiVec);
			this.AntiVec.normalize();
			this.AntiVec.multiplyScalar(-5);
			this.Object3D.position.add(this.AntiVec);				
		}
		
	}


// Функция тестирует акселерометр на показания данных
	testDeviceMotion = function (event) {
		// Если браузер поддерживает событие, но данные
		// передаются как undefined||null, значит, что устройство
		// не поддерживает акселерометр,
		// иначе данные были бы числом.
		if(event.rotationRate.alpha === null || event.rotationRate.alpha === undefined)
		{
			// Если у нас данные - не число, то удаляем все собственные обработчики;
			// чтобы они за зря не крутились
			window.removeEventListener("devicemotion", this.testDeviceMotion);	
			if(document.body.contains(this.AccelerometerControlButton))	
				document.body.removeChild(this.AccelerometerControlButton);
			if(document.body.contains(this.TouchControlButton))
				document.body.removeChild(this.TouchControlButton);
		} else {
			// если данные являются числом, то устройство имеет акселерометр,
			// устанавливаем нормальный обработчик и удаляем тестовый;
			this.update = this.accelerometerUpdate;
			window.removeEventListener("devicemotion", this.testDeviceMotion);			
			window.addEventListener("devicemotion", this.onDeviceMotion.bind(this));			
			window.addEventListener("deviceorientation", this.onDeviceOrientation.bind(this));			
		}
	}


// Наш обработчик ускорения
	onDeviceMotion(event) {

		this.OrientationParameters.alpha = event.rotationRate.alpha;
		this.OrientationParameters.beta = -event.rotationRate.beta;
		this.OrientationParameters.deviceMotionInterval = event.interval;

		this.OrientationParameters.phi = 0;
		this.OrientationParameters.theta = 0;
	}

// Наш обработчик ускорения
	onDeviceOrientation(event) {
		this.OrientationParameters.alpha = event.alpha;
		this.OrientationParameters.beta = -event.beta;
		this.OrientationParameters.deviceMotionInterval = event.interval;

		this.OrientationParameters.phi = 0;
		this.OrientationParameters.theta = 0;
	}
}