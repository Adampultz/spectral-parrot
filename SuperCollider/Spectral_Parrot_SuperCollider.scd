// Mock training file for sending 8 oscillators ala the parrot

(
var numChannels = 20; // Number of channels
var sampleRate = 44100; // Assume default sample rate (change if needed)
var createOSCHandler;


s.options.memSize_(2.pow(20));
s.options.sampleRate_(sampleRate);
s.options.numInputBusChannels_(numChannels);
s.options.numOutputBusChannels_(numChannels);
s.options.recChannels_(8);
// s.options.hardwareBufferSize = 64;
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";

s.options.outDevice_(
	// "BlackHole_16_Mac_Audio"
	"Fireface UFX III (24230602)"
);

s.options.inDevice_(
	// "BlackHole_16_Mac_Audio"
	"Fireface UFX III (24230602)"
);

~minFreq = 50;
~maxfreq = 5000;
~freqStepSize = 0.05;
~oscFreqs = 8.collect{exprand(~minFreq, ~maxfreq)}.sort;
~freqStates = 8.collect{1};

// ~testAudio
~oscFreqBusses = 8.collect({|i| Bus.control(s).set(~oscFreqs[i])});
~oscAmpBusses = 8.collect({|i| Bus.control(s, 1)});
~targetAudioBus = Bus.audio(s);
~inputAudioBus = Bus.audio(s);
~phasorTests = 8.collect{Bus.control(s).set(1)};
~directions = 8.collect{Bus.control(s).set(1)};
~ampDiffs = 8.collect{Bus.control(s).set(0.1)};
~sineStates = 8.collect{Bus.control(s).set(0.1)};
~stringGain = 8.collect{Bus.control(s).set(2.0)};

~mel_win_sizes = [128, 256, 1024, 2048];
~numMelBands = 40;
~visualize = true;

~pythonOSCAddress = NetAddr("127.0.0.1", 3333);  // Python's receiving port

OSCdef(\set_osc_params, {
	arg msg;
	var numOsc = 8;
	var freqCmd = msg[1..(numOsc)];  // First 8 values are frequencies
	var amps = msg[(numOsc+1)..(numOsc*2)];  // Next 8 values are amplitudes

	// Update oscillator frequencies
	freqCmd.do({ |cmd, i|
		var currentFreq = ~oscFreqs[i];
		var newFreq;

		if (cmd == -1) {
			// Decrease frequency by x% (e.g., 5%)
			newFreq = currentFreq / (1 + ~freqStepSize);
			// Apply lower boundary check
			newFreq = max(newFreq, ~minFreq);  // Minimum 50 Hz
		} {
			if (cmd == 1) {
				// Increase frequency by x% (e.g., 5%)
				newFreq = currentFreq * (1 + ~freqStepSize);
				// Apply upper boundary check
				newFreq = min(newFreq, ~maxfreq);  // Maximum 5000 Hz
			} {
				// Maintain current frequency
				newFreq = currentFreq;
			}
		};

		// Update frequency
		~oscFreqs[i] = newFreq;
		~oscFreqBusses[i].set(newFreq);
	});

	// Update oscillator amplitudes
	amps.do { |amp, i|
		if(i < numOsc) {
			~oscAmpBusses[i].set(amp);
			/*     if(0.5.coin) { // Only log occasionally to reduce clutter
			("Oscillator " ++ i ++ " amplitude set to " ++ amp).postln;
			};*/
		};
	};
}, "/set_osc_params");

// Handle reset message
OSCdef(\resetOscillators, {|msg, time, addr, recvPort|
	// Generate new random frequencies for all oscillators
	~oscFreqs = 8.collect{exprand(~minFreq, ~maxfreq)}.sort;

	// Apply the new frequencies
	~oscFreqs.do({ |freq, i|
		// Update your oscillator synths with the new frequencies
		~oscFreqBusses[i].set(freq);
	});

	"Oscillators reset with new random frequencies".postln;
}, '/reset_oscillators');

createOSCHandler = {|name, key, port|
	var running = False;
	OSCdef(name,
		{ |msg, time, addr, recvPort|
			// "OSCdef received:".postln;
			msg.postln;
			if (msg[1] == true, {
				if (running == False, {
					~trainingSynth = Synth(\trainingAudio, [\buf, ~trainingAudio, \analysisBus, ~targetAudioBus, \outChanMonitor, 18], target: ~audioGrp);
					/*~oscSynth = 8.collect{|i| Synth(\oscillators, [\freqBus, ~oscFreqBusses[i], \ampBus, ~oscAmpBusses[i], \analysisBus, ~inputAudioBus,  \outChanMonitor, 19], target: ~audioGrp)};*/

					~stringSynth = Synth(\strings, [\analysisBus, ~inputAudioBus, \stringGain, ~stringGain], target: ~audioGrp);
					~updateTask = Task({
						loop {
							~updateDirection.value;
							0.05.wait; // Update every 50ms
						}
					}).play;
				~analysisSynth = Synth(\analysis, [\inputAudioBus, ~inputAudioBus, \targetAudioBus, ~targetAudioBus], target: ~analysisGrp);
					running = True;
				})
			});
			if (running == True, {
				if (msg[1] == false,  {
					~trainingSynth.free;
					~oscSynth.do{|i|
						i.free}
				})
			})
		},
		key,
		nil,
		port
	).permanent_(true);
};

~win = Window("Mel Bands Monitor", Rect(10, 10, 620, 320)).front;
~ms = MultiSliderView(~win,
	Rect(0,0,~win.bounds.width,~win.bounds.height)
).elasticMode_(1).isFilled_(1);

OSCdef(\melBands,{
	arg msg;
	var melBands, numBands = msg[3];
	melBands = msg[4..(4+(numBands-1)).asInteger].linlin(-100.0, -10.0, 0.0, 1.0);
	~pythonOSCAddress.sendMsg("/test",  *melBands);
	if (~visualize, {
		defer{~ms.value_(melBands)};
	});
},"/melBands");

createOSCHandler.(\loss_osc, '/osc_from_python', 2222);

s.waitForBoot({

	~trainingAudio = Buffer.read(s, "/Users/adammac2023/Documents/Musik-business/Adam/2024/STIP-4/Audio/Training sets/Drones_summed.wav").normalize;

	~stringGain.do{|i| i.set(1.5)};

	s.sync;

	~audioGrp = Group.new;
	~analysisGrp = Group.new(~audioGrp, \addAfter);

	s.sync;

	~updateDirection = {
		8.do{|i|
			var phasorTestVal, ampDiffs, phasorDir, sineStates, sendVal, offSet = 0;
			// Get the phasor test value (0 or 1) for this string
			phasorTestVal = ~phasorTests[i].getSynchronous.sign;
			ampDiffs = ~ampDiffs[i].getSynchronous;

			// Convert phasorTest to -1/1 (direction multiplier)
			phasorDir = (phasorTestVal * 2) - 1;
			sineStates = ~sineStates[i].getSynchronous;

			if (phasorTestVal == 0,
				{phasorTestVal = 1});

			if (~freqStates[i] == 0, {
				~freqStates[i] = 1;
			});

			sendVal = phasorTestVal * ampDiffs * ~freqStates[i];

			~freqStates[i] = sendVal.sign.asInteger;


			/*        // If ampDiff and phasorDir have opposite signs, we need to reverse
			var shouldReverse = (ampDiff.sign * phasorDir) < 0;

			// Calculate effective direction
			var effectiveDir = ampDiff.sign;
			if(shouldReverse) { effectiveDir = effectiveDir.neg };*/

			// Set the new direction for this string
			~directions[i].set(sendVal);
		};
	};

	s.sync;

	SynthDef(\trainingAudio, {
		|buf, outChanPy, outChanMonitor, analysisBus|
		var src;
		// src = PlayBuf.ar(1, buf, loop: 1);
		src = 8.collect{
			var freq = 50.exprand(4950);
			var amp = 0.1.rrand(1).ampdb.linlin(-65, 1.0, 0.0, 1.0);
			SinOsc.ar(freq) * AmpComp.kr(freq) * amp * 0.1};
		src = Mix(src);
		Out.ar(outChanMonitor, src);
		Out.ar(analysisBus, src.scope);
	}).add;

	/*	SynthDef(\strings, {
	|freqBus, ampBus, outChanPy, outChanMonitor, analysisBus|
	var oscArray;
	// Use a control bus for frequency to allow real-time updates
	oscArray = SinOsc.ar(In.kr(freqBus)) * AmpComp.kr(In.kr(freqBus)) * In.kr(ampBus) * 0.1;
	Out.ar(outChanMonitor, oscArray);
	Out.ar(analysisBus, oscArray;
	}).add;*/

	SynthDef(\strings, {
		|freqBus, outChanPy, outChanMonitor, analysisBus, stringGain|
		var oscArray, amp, ampDiffNeighbours, directions, phasor, sineAM, phasorTest, sineState, sumSigs;

		// Input from 8 self-oscillating strings
		oscArray = SoundIn.ar([0, 1, 2, 3, 4, 5, 6, 7]);
		oscArray = oscArray * In.kr(stringGain).poll;

		// Get amplitude of each string, normalized
		// amp = Amplitude.ar(oscArray).ampdb.linlin(-70.0, -15.0, 0, 1.0);
		amp = RMS.ar(oscArray, 20).ampdb.linlin(-70.0, -15.0, 0, 1.0);
		amp = Integrator.ar(amp, 0.2);

		// Calculate the amplitude difference between a string and its neighbors
		ampDiffNeighbours = 8.collect{|i|
			var meanNeighbours, prevN, nextN;
			prevN = amp[(i - 1) % 8];
			nextN = amp[(i + 1) % 8];
			meanNeighbours = [prevN, nextN].mean;
			meanNeighbours - amp[i];
		};

		// Read direction values from control buses
		directions = 8.collect{|i| In.kr(~directions[i])};

		// Create phasors with rates based on amplitude differences and direction
		/*    phasor = 8.collect{|i|
		var rate = directions[i]  * SampleDur.ir;
		Phasor.ar(0, rate, 0, pi);
		};
		*/
		// Test if phasors are less than π/2 (for direction control)
		phasorTest = phasor.collect{|ph| ph < 0.5pi};


		// Create sine oscillators from phasors (0-1 range)
		sineAM = SinOsc.ar(directions, 0, 0.6, 0.45);
		sineState = sineAM > 0.5;

		// Output phasor tests to control buses
		8.do{|i| Out.kr(~phasorTests[i], Slope.ar(sineAM[i]))};
		8.do{|i| Out.kr(~ampDiffs[i], ampDiffNeighbours[i])};
		8.do{|i| Out.kr(~sineStates[i], sineState[i])};

		sumSigs = Mix(oscArray * sineAM);

		// Output mixed signal
		Out.ar([2, 3, 4, 5], sumSigs);
		Out.ar(analysisBus, sumSigs.scope);
	}).add;

	SynthDef(\analysis, {
		|inputAudioBus, targetAudioBus|
		var inputAudio, targetAudio, melBandsInput, melBandsTarget;
		inputAudio = In.ar(inputAudioBus);
		targetAudio = In.ar(targetAudioBus);
		melBandsInput = 4.collect{|i| FluidMelBands.kr(inputAudio, numBands: ~numMelBands, maxFreq: 10000,  scale: 1, windowSize: ~mel_win_sizes[i])};
		melBandsTarget = 4.collect{|i| FluidMelBands.kr(targetAudio, numBands: ~numMelBands, maxFreq: 10000,  scale: 1, windowSize: ~mel_win_sizes[i])};
		SendReply.ar(Impulse.ar((SampleRate.ir / 2048)),"/melBands", [~numMelBands * 2] ++ melBandsTarget.mean ++ melBandsInput.mean);
	}).add;

});
);

(
x = Synth(\strings, [\analysisBus, ~inputAudioBus, \stringGain, ~stringGain], target: ~audioGrp);
~analysisSynth = Synth(\analysis, [\inputAudioBus, ~inputAudioBus, \targetAudioBus, ~targetAudioBus], target: ~analysisGrp);
~updateTask = Task({
	loop {
		~updateDirection.value;
		0.05.wait; // Update every 50ms
	}
}).play;
)
s.plotTree

x.free
{In.kr(~stringGain).scope}.play
~analysisSynth.free


~stringGain[0].getSynchronous