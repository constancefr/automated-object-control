--------------------------------------------------------------------------------
-- Full specification of the nn_model networks


--------------------------------------------------------------------------------
-- Utilities


--------------------------------------------------------------------------------
-- Inputs

type Input = Tensor Real [4]

distanceToFrontCar = 0   
distanceToBackCar  = 1  
velocityToFrontCar = 2  
velocityToBackCar  = 3


--------------------------------------------------------------------------------
-- Outputs

type ActionToTake = Tensor Real [3]

brake      = 0
idle       = 1
accelerate = 2


--------------------------------------------------------------------------------
-- The network

@network
nnModel : Input -> ActionToTake

--------------------------------------------------------------------------------
-- Normalisation

type UnnormalisedInput = Tensor Real [4]

minimumInputValues : UnnormalisedInput
minimumInputValues = [-0.092464-0.1,-0.077204-0.1,-5.563068-0.1,-5.361264-0.1]

maximumInputValues : UnnormalisedInput
maximumInputValues = [311.600923+0.1, 305.828968+0.1, 7.548104+0.1, 7.422031+0.1]

validInput : UnnormalisedInput -> Bool
validInput x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i


meanScalingValues : UnnormalisedInput
meanScalingValues = [68.479297, 69.437695, 0.393321, 0.718784]

standardDeviation : UnnormalisedInput
standardDeviation = [52.128595, 52.274740, 2.142253, 1.982537]

normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / standardDeviation ! i

--------------------------------------------------------------------------------
-- Network with normalisation

normNnModel : UnnormalisedInput -> ActionToTake
normNnModel x = nnModel (normalise x)

actionToTake : Index 3 -> UnnormalisedInput -> Bool
actionToTake i x = forall j . 
    i != j => normNnModel x ! i <= normNnModel x ! j

--------------------------------------------------------------------------------
-- Definition of safety robustness around an action

-- This will means around a action point within epsilon ball
-- the NN network will always advise the same action

@parameter
epsilon : Real

-- This will control the PERTUBATION with in the epsilon ball
-- The pertubation will be used in function robustSafetAround
boundedByEpsilon : UnnormalisedInput -> Bool
boundedByEpsilon x = forall i . -epsilon <= x ! i  <= epsilon

-- This will check the robust safety around the action point
-- 
-- robustSafetyAround : UnnormalisedInput -> ActionToTake -> Bool
-- robustSafetyAround x action = forall pertubation .
--     let xPerturbed = x - pertubation in
--     (boundedByEpsilon pertubation) and validInput (xPerturbed) =>
--     actionToTake action xPerturbed

--------------------------------------------------------------------------------
-- Properties

Bmax = 5.0 -- magnitude
Amax = 3.0
Vmax = 20.0 -- ??
L = 4.0 -- car length

safeFront : UnnormalisedInput -> Bool
safeFront x =
  x ! distanceToFrontCar > L and
  x ! distanceToFrontCar > L + (Vmax * Vmax)/(2 * Bmax) 
-- Note: marabou can't handle non-linearity T_T so this is a very conservative estimate
-- Ideally we would have:
--   x ! distanceToFrontCar > L + (x ! velocityToFrontCar * x ! velocityToFrontCar)/(2 * Bmax)

safeBack : UnnormalisedInput -> Bool
safeBack x = 
  x ! distanceToBackCar > L and
  x ! distanceToFrontCar > L + (Vmax * Vmax)/(2 * Amax)
-- Non-linearly:
--   x ! distanceToBackCar > L + (x ! velocityToBackCar * x ! velocityToBackCar)/(2 * Amax)

-- If the back car is accelerating and too close, accelerate
@property
property1 : Bool
property1 = forall x .
  validInput x and
  not (safeFront x) =>
  actionToTake brake x

-- If the front car is braking and too close, brake
@property
property2 : Bool
property2 = forall x .
  validInput x and
  not (safeBack x) =>
  actionToTake accelerate x











