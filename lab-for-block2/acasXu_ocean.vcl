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
-- for the given input index(i) and model input (x)
-- Ture if the output(i) is the the minimal
-- minial means the model will advise this selection
minimal : Index 5 -> UnnormalisedInput -> Bool
minimal i x = forall j . i != j => normAcasXu x ! i < normAcasXu x ! j

-- Using this to normalise the model output
normaliseOutput : Output -> Output
normaliseOutput x = foreach i .
    (x ! i - 7.518884) / 373.94992

--------------------------------------------------------------------------------
-- property 1
-- If the intruder is distant (1) and
-- is significantly slower than the ownship (2),
-- the score of a COC advisory will always be 
-- below a certain fixed threshold (a).

distant : UnnormalisedInput -> Bool
distant x =
    x ! distanceToIntruder >= 55947.691

significantlySlowerThanOwnship : UnnormalisedInput -> Bool
significantlySlowerThanOwnship x =
    x ! speed         >= 1145 and
    x ! intruderSpeed <= 60

@property
property1 : Bool
property1 = forall x .
    (validInput x) and 
    (distant x) and 
    (significantlySlowerThanOwnship x) =>
    (normaliseOutput(normAcasXu x) ! clearOfConflict >= 1500)

--------------------------------------------------------------------------------
-- Property 2
-- If the intruder is distant (1) and,
-- is significantly slower than the ownship (2),
-- the score of a COC advisory will never be maximal (b).

-- for the given input index(i) and model input (x)
-- Ture if the output(i) is the the minimal
maximal : Index 5 -> UnnormalisedInput -> Bool
maximal i x = forall j . i != j => normAcasXu x ! i > normAcasXu x ! j

@property
property2 : Bool
property2 = forall x .
    (validInput x) and 
    (distant x) and 
    (significantlySlowerThanOwnship x) =>
    not (maximal clearOfConflict x)

--------------------------------------------------------------------------------
-- Property 3

-- If the intruder is directly ahead (3) and
-- is moving towards the ownship (4),
-- the score for COC will not be minimal (c).


directlyAhead : UnnormalisedInput -> Bool
directlyAhead x =
    1500  <= x ! distanceToIntruder <= 1800 and
    -0.06 <= x ! angleToIntruder    <= 0.06

movingTowards : UnnormalisedInput -> Bool
movingTowards x =
    x ! intruderHeading >= 3.10  and
    x ! speed           >= 980   and
    x ! intruderSpeed   >= 960

@property
property3 : Bool
property3 = forall x .
    (validInput) x and 
    (directlyAhead) x and 
    (movingTowards) x =>
    not (minimal clearOfConflict x)

--------------------------------------------------------------------------------
-- Property 4

-- If the intruder is directly ahead (3) and
-- is moving away from the ownship (5) but
-- at a lower speed than that of the ownship (6), 
-- the score for COC will not be minimal (c).

movingAway : UnnormalisedInput -> Bool
movingAway x = 
    x ! intruderHeading == 0

lowerSpeedThanOwnship : UnnormalisedInput -> Bool
lowerSpeedThanOwnship x =
    1000 <= x ! speed and
    700  <= x ! intruderSpeed <= 800

@property
property4 : Bool
property4 = forall x .
    (validInput x) and 
    (directlyAhead x) and 
    (movingAway x) and
    (lowerSpeedThanOwnship x) =>
    not (minimal clearOfConflict x)

--------------------------------------------------------------------------------
-- Property 5

-- If the intruder is near (6) and 
-- approaching from the left (7), 
-- the network advises strong right (d)

near : UnnormalisedInput -> Bool
near x =
    250 <= x ! distanceToIntruder <= 400

movingFromLeft : UnnormalisedInput -> Bool
movingFromLeft x =
    0.2       <= x ! angleToIntruder <= 0.4 and
    -3.141592 <= x ! intruderHeading <= -3.141592 + 0.0005 and
    100       <= x ! speed           <= 400 and
    0         <= x ! intruderSpeed   <= 400

@property
property5 : Bool
property5 = forall x .
    (validInput x) and 
    (near x) and 
    (movingFromLeft x) =>
    (minimal strongLeft x)

--------------------------------------------------------------------------------
-- Property 6

-- If the intruder is sufficiently far away (8)
-- the network advises COC (e)

sufficientlyFarAway : UnnormalisedInput -> Bool
sufficientlyFarAway x =
    12000     <= x ! distanceToIntruder <= 62000 and
    (0.7      <= x ! angleToIntruder    <= 3.141592 or 
    -3.141592 <= x ! angleToIntruder    <= -0.7) and
    -3.141592 <= x ! intruderHeading    <= -3.141592 + 0.0005 and
    100       <= x ! speed              <= 1200 and
    0         <= x ! intruderSpeed      <= 1200

@property
property6 : Bool
property6 = forall x .
    (validInput x) and 
    (sufficientlyFarAway x) =>
    (minimal clearOfConflict x)


--------------------------------------------------------------------------------
-- Property 7

-- If vertical separation is large (9), 
-- the network will never advise a strong turn (f).

verticalSeparationLarge : UnnormalisedInput -> Bool
verticalSeparationLarge x =
    0         <= x ! distanceToIntruder <= 60760 and
    -3.141592 <= x ! angleToIntruder    <= 3.141592 and
    -3.141592 <= x ! intruderHeading    <= 3.141592 and
    100       <= x ! speed              <= 1200 and
    0         <= x ! intruderSpeed      <= 1200

@property
property7 : Bool
property7 = forall x .
    (validInput x) and 
    (verticalSeparationLarge x) =>
    not (minimal strongLeft x) and 
    not (minimal strongRight x)

--------------------------------------------------------------------------------
-- Property 8

-- For a large vertical separation (9) and
-- a previous “weak left” advisory (10), 
-- the network will either output COC (e) or 
-- continue advising “weak left” (f).

largeVerticalSeparationAndPreviousWeakLeft : UnnormalisedInput -> Bool
largeVerticalSeparationAndPreviousWeakLeft x =
    0         <= x ! distanceToIntruder <= 60760 and
    -3.141592 <= x ! angleToIntruder    <= -0.75 * 3.141592 and
    -0.1      <= x ! intruderHeading    <= 0.1 and
    600       <= x ! speed              <= 1200 and
    600       <= x ! intruderSpeed      <= 1200 and
    (minimal weakLeft x)

@property
property8 : Bool
property8 = forall x .
    (validInput x) and 
    (largeVerticalSeparationAndPreviousWeakLeft x) =>
    (minimal clearOfConflict x) or
    (minimal weakLeft x)

--------------------------------------------------------------------------------
-- Property 9

-- Even if the previous advisory was “weak right” (11), 
-- the presence of a nearby intruder (12) 
-- will cause the network to output a “strong left” advisory instead.

nearbyIntruderAndPreviousWeakRight : UnnormalisedInput -> Bool
nearbyIntruderAndPreviousWeakRight x =
    2000      <= x ! distanceToIntruder <= 7000 and
    -0.4      <= x ! angleToIntruder    <= -0.14 and
    -3.141592 <= x ! intruderHeading    <= -3.141592 + 0.01 and
    100       <= x ! speed              <= 150 and
    0         <= x ! intruderSpeed      <= 150 

@property
property9 : Bool
property9 = forall x .
    (validInput x) and 
    (nearbyIntruderAndPreviousWeakRight x) =>
    (minimal strongLeft x)

--------------------------------------------------------------------------------
-- Property 10

-- For a far away intruder (13), 
-- the network advises COC (e)

farAwayIntruder : UnnormalisedInput -> Bool
farAwayIntruder x =
    36000     <= x ! distanceToIntruder <= 60760 and
    0.7       <= x ! angleToIntruder    <= 3.141592 and
    -3.141592 <= x ! intruderHeading    <= -3.141592 + 0.01 and
    900       <= x ! speed              <= 1200 and
    600       <= x ! intruderSpeed      <= 1200

@property
property10 : Bool
property10 = forall x .
    (validInput x) and 
    (nearbyIntruderAndPreviousWeakRight x) =>
    (minimal clearOfConflict x)


















