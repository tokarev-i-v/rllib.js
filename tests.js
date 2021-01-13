/**
 * Testing PPO algorithm executor.
 */

var Jasmine = require('jasmine');
var jasmine = new Jasmine();

//jasmine.execute(["agents/AC/test_AC.js", "Testing Actor-Critic"]);
jasmine.execute(["agents/policy_gradients/test_PPO.js", "Testing PPO"]);