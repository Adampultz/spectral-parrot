(
var numChannels = 20;
var sampleRate = 48000;
var createOSCHandler;

s.options.memSize_(2.pow(20));
s.options.sampleRate_(sampleRate);
s.options.numInputBusChannels_(30);
s.options.hardwareBufferSize_(64);
s.options.numOutputBusChannels_(30);
s.options.recChannels_(8);
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";

s.options.outDevice_("Blackhole16_FirefaceUFXIII");
s.options.inDevice_("Blackhole16_FirefaceUFXIII");

~inChannel = 24;
~outInAudio = 0;
~outTargetAudio = 1;
~stringInChan = 16;
~parrotOutChan = 18;
~trainingOutChan = 16;
~outBlackholeTraining = 0;
~outBlackholeStrings = 1;
~currentTrainingAudioIndex = 0;
~delayTime = 0.005; // to compensate for phase issues between the two speakers

~minFreq = 40;
~maxfreq = 2000;
~oscFreqs = 8.collect{exprand(~minFreq, ~maxfreq)}.sort;
~freqStates = 8.collect{1};

~targetAudioBus = Bus.audio(s);
~inputAudioBus = Bus.audio(s);
~phasorTests = 8.collect{Bus.control(s).set(1)};
~directions = 8.collect{Bus.control(s).set(1)};
~ampDiffs = 8.collect{Bus.control(s).set(0.1)};
~sineStates = 8.collect{Bus.control(s).set(0.1)};
~stringGain = 8.collect{Bus.control(s).set(1.0)};
~trainingAudioSel = Bus.control(s).set(1);  // 0 = Audio input, 1 = Sample, 2 = harmonic sine spectrums, 3 = Summed sines (random frequencies),
~integration_factor = Bus.control(s).set(0.3);
~delayTimeBus = Bus.control(s);

// Initialize as nil - will be populated by OSC message
~trainingAudioPaths = nil;
~trainingAudioBuffers = nil;
~trainingAudio = nil;  // Current active buffer

~pythonOSCAddress = NetAddr("127.0.0.1", 3333);

~initiate = {
	// Make sure we have loaded audio before starting
	if (~trainingAudio.isNil, {
		"ERROR: No training audio loaded! Send /load_training_audio_path first.".postln;
	}, {
		~trainingSynth = Synth(\trainingAudio, [
			\buf, ~trainingAudio,  // Use ~trainingAudio directly (current buffer)
			\analysisBus, ~targetAudioBus,
			\trainingAudioSel, ~trainingAudioSel,
			\inChanOffset, ~inChannel,
			\outChanMonitor, ~trainingOutChan,
			\outBlackhole, ~outBlackholeTraining
		], target: ~audioGrp);

		~stringSynth = Synth(\strings, [
			\analysisBus, ~inputAudioBus,
			\inChanOffset, ~stringInChan,
			\stringGain, ~stringGain,
			\outChan, ~parrotOutChan,
			\outBlackhole, ~outBlackholeStrings,
			\delayTime, ~delayTimeBus,
		], target: ~audioGrp);

		~updateTask = Task({
			loop {
				~updateDirection.value;
				0.005.wait;
			}
		}).play;

		"Training synths started successfully".postln;
	});
};

// OSCdef to load training audio from a folder path
OSCdef(\loadTrainingAudioPath, {|msg, time, addr, recvPort|
	var audioPath = msg[1].asString;
	var rotationCount = msg[2].asInteger;  // Get rotation count from Python
	var pathObj = PathName(audioPath);
	var audioFiles;
	var startIndex;

	"=== Loading Training Audio ===".postln;
	("Folder path: " ++ audioPath).postln;
	("Rotation count: " ++ rotationCount).postln;

	// Get all audio files from the folder (wav, aiff, aif)
	audioFiles = pathObj.files.select({ |file|
		var ext = file.extension.toLower;
		(ext == "wav") || (ext == "aiff") || (ext == "aif")
	});

	if (audioFiles.size == 0, {
		"ERROR: No audio files found in folder!".postln;
		~pythonOSCAddress.sendMsg("/training_audio_loaded", 0, 0, "No audio files found");
	}, {
		("Found " ++ audioFiles.size ++ " audio files").postln;

		// Calculate starting index using modulo
		startIndex = rotationCount % audioFiles.size;
		("Calculated starting index: " ++ startIndex).postln;

		// Store the paths
		~trainingAudioPaths = audioFiles;

		// Allocate array for buffers
		~trainingAudioBuffers = Array.newClear(audioFiles.size);

		// Load all buffers
		audioFiles.do({ |file, i|
			("Loading [" ++ (i+1) ++ "/" ++ audioFiles.size ++ "]: " ++ file.fileName).postln;

			~trainingAudioBuffers[i] = Buffer.read(s, file.fullPath, action: { |buf|
				if (buf.notNil, {
					buf.normalize;
					("✓ Loaded: " ++ file.fileName).postln;
				}, {
					("ERROR loading: " ++ file.fileName).postln;
				});
			});
		});

		// Wait for all buffers to load, then set the CORRECT starting buffer
		fork {
			s.sync;  // Wait for all buffer operations to complete

			// Set the starting index from Python's calculation
			~currentTrainingAudioIndex = startIndex;
			~trainingAudio = ~trainingAudioBuffers[startIndex];

			"=== All Training Audio Loaded Successfully ===".postln;
			("Total buffers: " ++ ~trainingAudioBuffers.size).postln;
			("Active buffer: #" ++ (startIndex + 1) ++ " - " ++ ~trainingAudioPaths[startIndex].fileName).postln;

			// Send confirmation to Python WITH the starting index
			~pythonOSCAddress.sendMsg("/training_audio_loaded", ~trainingAudioBuffers.size, startIndex, "success");
		};
	});
}, '/load_training_audio_path');

// OSCdef to switch between pre-loaded buffers
OSCdef(\switchTrainingAudio, {|msg, time, addr, recvPort|
	var newIndex = msg[1].asInteger;

	if (~trainingAudioBuffers.isNil, {
		"ERROR: No training audio buffers loaded yet!".postln;
		~pythonOSCAddress.sendMsg("/training_audio_switched", newIndex, 0, "No buffers loaded");
	}, {
		if ((newIndex >= 0) && (newIndex < ~trainingAudioBuffers.size), {
			~currentTrainingAudioIndex = newIndex;
			~trainingAudio = ~trainingAudioBuffers[newIndex];

			("Switched to buffer [" ++ (newIndex + 1) ++ "/" ++ ~trainingAudioBuffers.size ++ "]: " ++ ~trainingAudioPaths[newIndex].fileName).postln;

			// Update the running synth if it exists
			if (~trainingSynth.notNil, {
				~trainingSynth.set(\buf, ~trainingAudio);
				"Updated synth with new buffer".postln;
			});

			// Send confirmation back to Python
			~pythonOSCAddress.sendMsg("/training_audio_switched", newIndex, 1, "success");
		}, {
			("ERROR: Invalid buffer index " ++ newIndex ++ " (must be 0-" ++ (~trainingAudioBuffers.size - 1) ++ ")").postln;
			~pythonOSCAddress.sendMsg("/training_audio_switched", newIndex, 0, "Invalid index");
		});
	});
}, '/switch_training_audio');

// OSCdef to query the number of loaded buffers
OSCdef(\getNumTrainingAudios, {|msg, time, addr, recvPort|
	var numBuffers = if (~trainingAudioBuffers.notNil, { ~trainingAudioBuffers.size }, { 0 });
	~pythonOSCAddress.sendMsg("/num_training_audios", numBuffers);
	("Sent buffer count to Python: " ++ numBuffers).postln;
}, '/get_num_training_audios');

// OSCdef for resetting oscillators
OSCdef(\resetOscillators, {|msg, time, addr, recvPort|
	~oscFreqs = 8.collect{exprand(~minFreq, ~maxfreq)}.sort;
	~oscFreqs.do({ |freq, i|
		~oscFreqBusses[i].set(freq);
	});
	"Oscillators reset with new random frequencies".postln;
}, '/reset_oscillators');

// Start/stop handler
createOSCHandler = {|name, key, port|
	var running = false;
	OSCdef(name,
		{ |msg, time, addr, recvPort|
			msg.postln;

			if (msg[1] == true, {
				if (running == false, {
					{
						~initiate.value;
						s.sync;
						running = true;
					}.fork;
				})
			});

			if (msg[1] == false, {
				if (running == true, {
					"SuperCollider shutting down synths".postln;
					{
						if (~trainingSynth.notNil, { ~trainingSynth.free });
						if (~stringSynth.notNil, { ~stringSynth.free });
						if (~updateTask.notNil, { ~updateTask.stop });
						s.sync;
						running = false;
					}.fork;
				})
			})
		},
		key, nil, port
	).permanent_(true);
};

createOSCHandler.(\loss_osc, '/osc_from_python', 2222);

s.waitForBoot({

	8.do{|i|

		~phasorTests[i].set(1);
		~directions[i].set(1);
		~ampDiffs[i].set(0.1);

		~sineStates[i].set(0.1);
		~stringGain[i].set(1.5);
	};
	~trainingAudioSel.set(1);  // 0 = Audio input, 1 = Sample, 2 = harmonic sine spectrums, 3 = Summed sines (random frequencies),
	~integration_factor.set(0.3);

	// Set string gains
	~stringGain.do{|i| i.set(1.5)};
	~delayTimeBus.set(~delayTime);

	s.sync;

	"SuperCollider booted - waiting for training audio path from Python...".postln;

	// Create audio groups
	~audioGrp = Group.new;
	~analysisGrp = Group.new(~audioGrp, \addAfter);

	s.sync;

	// Define motor direction update function
	~updateDirection = {
		8.do{|i|
			var phasorTestVal, ampDiffs, phasorDir, sineStates, sendVal;

			phasorTestVal = ~phasorTests[i].getSynchronous.sign;
			ampDiffs = ~ampDiffs[i].getSynchronous;
			sineStates = ~sineStates[i].getSynchronous;

			if (phasorTestVal == 0, { phasorTestVal = 1 });
			if (~freqStates[i] == 0, { ~freqStates[i] = 1 });

			phasorDir = (phasorTestVal * 2) - 1;
			sendVal = phasorTestVal * ampDiffs * ~freqStates[i];
			~freqStates[i] = sendVal.sign.asInteger;

			~directions[i].set(sendVal);
		};
	};

	s.sync;

	// Define SynthDefs
	SynthDef(\trainingAudio, {
		|buf, outChanPy, outChanMonitor, analysisBus, inChanOffset, trainingAudioSel, outBlackhole|
		var src;

		src = Select.ar(In.kr(trainingAudioSel),
			[
				// Mode 0: Audio input
				Mix.ar(SoundIn.ar([inChanOffset, inChanOffset + 1, inChanOffset + 2, inChanOffset + 3])),

				// Mode 1: Buffer playback
				PlayBuf.ar(1, buf, loop: 1) * 0.5,

				// Mode 2: Harmonic sine spectrum
				Mix(SinOsc.ar([30, 60, 90, 120, 150, 180, 210, 240], 0,
					[1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.025, 0.0125])) * 0.05,

				// Mode 3: Random sine mixture
				Mix(8.collect{
					var freq = ~minFreq.exprand(~maxfreq);
					var amp = 0.1.rrand(1).ampdb.linlin(-65, 1.0, 0.0, 1.0);
					SinOsc.ar(freq) * AmpComp.kr(freq) * amp * 0.05
				})
			]
		);

		Out.ar(outBlackhole, src);
		Out.ar(outChanMonitor, src);
	}).add;

	SynthDef(\strings, {
		|freqBus, inChanOffset, outChanPy, outChanMonitor, analysisBus, stringGain, outChan, outBlackhole, delayTime|
		var oscArray, amp, ampDiffNeighbours, directions, sineAM, sineState, sumSigs, sumSigsDelayed;

		// Input from 8 self-oscillating strings
		oscArray = SoundIn.ar(8.collect{|i| inChanOffset + i});
		oscArray = oscArray * In.kr(stringGain);

		// Get amplitude of each string
		amp = RMS.ar(oscArray, 20).ampdb.linlin(-65.0, 0.0, 0, 1.0);
		amp = Integrator.ar(amp, In.kr(~integration_factor)).clip(0.0, 1.0);

		// Calculate amplitude difference between string and neighbors
		ampDiffNeighbours = 8.collect{|i|
			var meanNeighbours, prevN, nextN;
			prevN = amp[(i - 1) % 8];
			nextN = amp[(i + 1) % 8];
			meanNeighbours = [prevN, nextN].mean;
			meanNeighbours - amp[i];
		};

		// Read direction values from control buses
		directions = 8.collect{|i| In.kr(~directions[i])};

		// Create sine AM oscillators
		sineAM = SinOsc.ar(directions, 0, 0.6, 0.45);
		sineState = sineAM > 0.5;

		// Output to control buses
		8.do{|i| Out.kr(~phasorTests[i], Slope.ar(sineAM[i]))};
		8.do{|i| Out.kr(~ampDiffs[i], ampDiffNeighbours[i])};
		8.do{|i| Out.kr(~sineStates[i], sineState[i])};

		sumSigs = Mix(LeakDC.ar(oscArray * sineAM)) * 3.0;
		sumSigsDelayed = DelayN.ar(sumSigs, delaytime: In.kr(delayTime));

		// Output mixed signal
		Out.ar(outChan, [sumSigs, sumSigsDelayed]);
		Out.ar(outBlackhole, Mix(oscArray));
	}).add;

	s.sync;

	"SuperCollider ready! Send /load_training_audio_path to begin.".postln;
});
);

s.plotTree