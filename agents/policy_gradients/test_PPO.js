import * as tf from '@tensorflow/tfjs-node-gpu';

import {Buffer} from "./PPO";

describe("PPO Buffer testing", function(){
    let b = new Buffer();
    it("and so is a spec", function() {
        b.store([
            [10,1,2,4,5],
            [10,1,2,4,5],
            [10,1,2,4,5],
            [10,1,2,4,5],
        ], 0);
    
        expect(a).toBe(true);
      });
});