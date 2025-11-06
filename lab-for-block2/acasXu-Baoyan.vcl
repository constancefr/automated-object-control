--------------------------------------------------------------------------------
-- Full specification of the ACAS XU networks

-- Taken from Appendix VI of "Reluplex: An Efficient SMT Solver for Verifying
-- Deep Neural Networks" at https://arxiv.org/pdf/1702.01135.pdf

-- Comments describing the properties are taken directly from the text.

--------------------------------------------------------------------------------
-- Utilities

-- The value of the constant `pi`.
pi = 3.141592

--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a vector of 5 rational numbers.

type Input = Tensor Real [5]

-- Next we add meaningful names for the indices.
-- The fact that all vector types come annotated with their size means that it
-- is impossible to mess up indexing into vectors, e.g. if you changed
-- `distanceToIntruder = 0` to `distanceToIntruder = 5` the specification would
-- fail to type-check.

distanceToIntruder = 0   -- measured in metres
angleToIntruder    = 1   -- measured in radians
intruderHeading    = 2   -- measured in radians
speed              = 3   -- measured in metres/second
intruderSpeed      = 4   -- measured in meters/second

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are also a vector of 5 rationals. Each one representing the score
-- for the 5 available courses of action.

type Output = Tensor Real [5]

-- Again we define meaningful names for the indices into output vectors.

clearOfConflict = 0
weakLeft        = 1
weakRight       = 2
strongLeft      = 3
strongRight     = 4

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
acasXu : Input -> Output

--------------------------------------------------------------------------------
-- Normalisation

-- As is common in machine learning, the network operates over
-- normalised values, rather than values in the problem space
-- (e.g. using standard units like m/s).
-- This is an issue for us, as we would like to write our specification in
-- terms of the problem space values .
-- Therefore before applying the network, we first have to normalise
-- the values in the problem space.

-- For clarity, we therefore define a new type synonym
-- for unnormalised input vectors which are in the problem space.
type UnnormalisedInput = Tensor Real [5]

-- Next we define the minimum and maximum values that each input can take.
-- These correspond to the range of the inputs that the network is designed
-- to work over.
minimumInputValues : UnnormalisedInput
minimumInputValues = [0,0,0,0,0]

maximumInputValues : UnnormalisedInput
maximumInputValues = [60261.0, 2*pi, 2*pi, 1100.0, 1200.0]

-- We can therefore define a simple predicate saying whether a given input
-- vector is in the right range.
validInput : UnnormalisedInput -> Bool
validInput x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i

-- Then the mean values that will be used to scale the inputs.
meanScalingValues : UnnormalisedInput
meanScalingValues = [19791.091, 0.0, 0.0, 650.0, 600.0]

-- We can now define the normalisation function that takes an input vector and
-- returns the unnormalised version.
normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / (maximumInputValues ! i)

-- Using this we can define a new function that first normalises the input
-- vector and then applies the neural network.
normAcasXu : UnnormalisedInput -> Output
normAcasXu x = acasXu (normalise x)

-- A constraint that says the network chooses output `i` when given the
-- input `x`. We must necessarily provide a finite index that is less than 5
-- (i.e. of type Index 5). The `a ! b` operator lookups index `b` in vector `a`.
advises : Index 5 -> UnnormalisedInput -> Bool
advises i x = forall j . i != j => normAcasXu x ! i < normAcasXu x ! j


directlyAhead : UnnormalisedInput -> Bool
directlyAhead x = 
	1500 <= x ! distanceToIntruder <= 1800 and
	-0.06 <= x ! angleToIntruder <= 0.06

movingTowards : UnnormalisedInput -> Bool
movingTowards x = 
	x ! intruderHeading >= 3.10 and 
	x ! speed >= 980 and
	x ! intruderSpeed >= 960

--COC is the maximal when its score is greater than all 4 others
cocMaximal : Index 5 -> UnnormalisedInput -> Bool
cocMaximal i x = forall j . j != i => normAcasXu x ! i > normAcasXu x ! j

--------------------------------------------------------------------------------
-- Property 3 - Do it yourself!

-- If the intruder is directly ahead and is moving towards the
-- ownship, the score for COC will not be minimal.

-- Tested on: all networks except N_{1,7}, N_{1,8}, and N_{1,9}.

@property
property3 : Bool
property3 = forall x .
	validInput x and directlyAhead x and movingTowards x =>
	not (advises clearOfConflict x)

--------------------------------------------------------------------------------

-- Property 1
-- Description: If the intruder is distant and is significantly slower than the
-- ownership, the score of a COC advisory will always be below a certain fixed
-- threshold.
-- Tested on: all 45 networks.
-- Input constraints: ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60.
-- Desired output property: the score for COC is at most 1500.

@property
property1 : Bool
property1 = forall x .
	validInput x and 
	x ! speed >= 1145 and 
	x ! intruderSpeed <= 60 and 
	x ! distanceToIntruder >= 55947.691 =>
	normAcasXu x ! 	clearOfConflict <= 1500

--------------------------------------------------------------------------------

-- Property 2
-- Description: If the intruder is distant and is significantly slower than the
-- ownship, the score of a COC advisory will never be maximal.
-- Tested on: Nx,y for all x ≥ 2 and for all y.
-- Input constraints: ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60.
-- Desired output property: the score for COC is not the maximal score.

@property
property2 : Bool
property2 = forall x . 
	validInput x and 
	x ! speed >= 1145 and 
	x ! intruderSpeed <= 60 and 
	x ! distanceToIntruder >= 55947.691 and
	x ! angleToIntruder  == 0 =>
	not (cocMaximal clearOfConflict x)

--------------------------------------------------------------------------------

-- Property 4
-- Description: If the intruder is directly ahead and is moving away from the
-- ownship but at a lower speed than that of the ownship, the score for COC
-- will not be minimal.
-- Tested on: all networks except N1,7, N1,8, and N1,9.
-- Input constraints: 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ = 0, vown ≥ 1000,
-- 700 ≤ vint ≤ 800.
-- Desired output property: the score for COC is not the minimal score.

@property
property4 : Bool
property4 = forall x . 
	validInput x and 
	directlyAhead x and
	x ! speed >= 1000 and
	700 <= x ! intruderSpeed <= 800 and 
	x ! intruderHeading == 0 =>
	not (advises clearOfConflict x)

--------------------------------------------------------------------------------

-- Property φ5.
-- Description: If the intruder is near and approaching from the left, the network
-- advises “strong right”.
-- Tested on: N1,1.
-- Input constraints: 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, −3.141592 ≤ ψ ≤
-- −3.141592 + 0.005, 100 ≤ vown ≤ 400, 0 ≤ vint ≤ 400.
-- Desired output p

@property
property5: Bool
property5 = forall x . 
	validInput x and
	250 <= x ! distanceToIntruder <= 400 and
	0.2 <= x ! angleToIntruder <= 0.4 and
	-pi <= x ! intruderHeading <= -pi + 0.005 and
	100 <= x ! speed <= 400 and
	0 <= x ! intruderSpeed <= 400 => 
	advises strongRight x

--------------------------------------------------------------------------------

-- Property φ6.
-- Description: If the intruder is sufficiently far away, the network advises COC.
-- Tested on: N1,1.
-- Input constraints: 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤
-- θ ≤ −0.7), −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 1200,
-- 0 ≤ vint ≤ 1200.
-- Desired output property: the score for COC is the minimal score.

@property
property6 : Bool
property6 = forall x .
	validInput x and
	12000 <= x ! distanceToIntruder <= 62000 and
	(
		(0.7 <= x ! angleToIntruder <= pi) or (-pi <= x ! angleToIntruder <= -0.7)
	) and
	-pi <= x ! intruderHeading <= -pi + 0.005 and
	100 <= x ! speed <= 1200 and
	0 <= x ! intruderSpeed <= 1200 =>
	advises clearOfConflict x

--------------------------------------------------------------------------------

-- Property φ7.
-- Description: If vertical separation is large, the network will never advise a
-- strong turn.
-- Tested on: N1,9.
-- Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤
-- ψ ≤ 3.141592, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
-- Desired output property: the scores for “strong right” and “strong left” are
-- never the minimal scores

@property
property7 : Bool
property7 = forall x .
	validInput x and
	0 <= x ! distanceToIntruder <= maximumInputValues ! distanceToIntruder and
	-pi <= x ! angleToIntruder <= pi and
	-pi <= x ! intruderHeading <= pi and
	100 <= x ! speed <= maximumInputValues ! speed and
	0 <= x ! intruderSpeed <= maximumInputValues ! intruderSpeed =>
	(not (advises strongRight x)) and (not (advises strongLeft x))

--------------------------------------------------------------------------------

-- Property φ8.
-- Description: For a large vertical separation and a previous “weak left” advisory, the network will either 
-- output COC or continue advising “weak left”.
-- Tested on: N2,9.
-- Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ −0.75·3.141592, −0.1 ≤
-- ψ ≤ 0.1, 600 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
-- Desired output property: the score for “weak left” is minimal or the score
-- for COC is minimal.

@property
property8 : Bool
property8 = forall x . 
	validInput x and 
	0 <= x ! distanceToIntruder <= maximumInputValues ! distanceToIntruder and 
	-pi <= x ! angleToIntruder <= (-0.75*pi) and
	-0.1 <= x ! intruderHeading <= 0.1 and
	600 <= x ! speed <= maximumInputValues ! speed and
	600 <= x ! intruderSpeed <= maximumInputValues ! intruderSpeed =>
	(advises weakLeft x or advises clearOfConflict x)
	

--------------------------------------------------------------------------------

-- Property φ9.
-- Description: Even if the previous advisory was “weak right”, the presence of
-- a nearby intruder will cause the network to output a “strong left” advisory
-- instead.
-- Tested on: N3,3.
-- Input constraints: 2000 ≤ ρ ≤ 7000, −0.4 ≤ θ ≤ −0.14, −3.141592 ≤ ψ ≤
-- −3.141592 + 0.01, 100 ≤ vown ≤ 150, 0 ≤ vint ≤ 150.
-- Desired output property: the score for “strong left” is minimal.

@property
property9 : Bool
property9 = forall x . 
	validInput x and 
	2000 <= x ! distanceToIntruder <= 7000 and
	-0.4 <= x ! angleToIntruder <= -0.14 and
	-pi <= x ! intruderHeading <= (-pi + 0.01) and
	100 <= x ! speed <= 150 and
	0 <= x ! intruderSpeed <= 150 =>
	advises strongLeft x


--------------------------------------------------------------------------------

-- Property φ10.
-- Description: For a far away intruder, the network advises COC.
-- Tested on: N4,5.
-- Input constraints: 36000 ≤ ρ ≤ 60760, 0.7 ≤ θ ≤ 3.141592, −3.141592 ≤
-- ψ ≤ −3.141592 + 0.01, 900 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
-- Desired output property: the score for COC is minimal.

@property
property10 : Bool
property10 = forall x .
	validInput x and 
	36000 <= x ! distanceToIntruder <= maximumInputValues ! distanceToIntruder and
	0.7 <= x ! angleToIntruder <= pi and
	-pi <= x ! intruderHeading <= (-pi + 0.01) and
	900 <= x ! speed <= maximumInputValues ! speed and
	600 <= x ! intruderSpeed <= maximumInputValues ! intruderSpeed =>
	advises clearOfConflict x

	