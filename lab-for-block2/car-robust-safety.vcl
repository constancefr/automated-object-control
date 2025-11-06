--------------------------------------------------------------------------------
-- Full specification of the nn_model networks


--------------------------------------------------------------------------------
-- Utilities
L = 0.01

@parameter
eta: Real

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

break      = 0
idle       = 1
accelerate = 2


--------------------------------------------------------------------------------
-- The network

@network
nnModel : Input -> ActionToTake

type Label = Index 3



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

-- This is the adviser
-- The dataset is normalised, so directly goes to nnModel
actionToTake : Index 3 -> Input -> Bool
actionToTake i x = forall j . 
    i != j => nnModel x ! i >= nnModel x ! j

-- This gives f(x)_j for every (j != i)
smallerThanEta : Index 3 -> Input -> Bool
smallerThanEta i x = forall j .
    i != j => nnModel x ! j <= eta


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


-- boundedByEpsilonEuclidean : UnnormalisedInput -> Bool
-- boundedByEpsilonEuclidean x = forall i .
    -- (x!i)*(x!i) <= eta*eta


-- This will check the robust safety around the action point
-- 
robustSafetyAround : Input -> Label -> Bool
robustSafetyAround x action = forall pertubation .
    let xPerturbed = x - pertubation in
    (boundedByEpsilon pertubation) and validInput (xPerturbed) =>
    actionToTake action xPerturbed

strongRobustSafetyAround : Input -> Label -> Bool
strongRobustSafetyAround x action = forall pertubation .
    let xPerturbed = x - pertubation in
    (boundedByEpsilon pertubation) and validInput (xPerturbed) =>
    smallerThanEta action xPerturbed

-- lipschitzRobustSafetyAround : Input -> Label -> Bool
-- lipschitzRobustSafetyAround x action = forall pertubation .
--     let xPerturbed = x - pertubation in
--     (boundedByEpsilon pertubation) and validInput (xPerturbed) =>
--     -L <= (nnModel x ! i) - (nnModel xPerturbed ! ) <= L
 

-- robustSafetyAroundEuclidean : Input -> Label -> Bool
-- robustSafetyAroundEuclidean x action = forall pertubation .
--     let xPerturbed = x - pertubation in
--     (boundedByEpsilonEuclidean pertubation) and validInput (xPerturbed) =>
--     actionToTake action xPerturbed

--------------------------------------------------------------------------------
-- Robustness with respect to a dataset

-- This is auto given parameter to make sure 
-- the input and labels are in the same length
@parameter(infer=True)
n : Nat

@dataset
trainingInputs : Vector Input n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . 
    robustSafetyAround (trainingInputs ! i) (trainingLabels ! i)


@property
strongRobust : Vector Bool n
strongRobust = foreach i . 
    strongRobustSafetyAround (trainingInputs ! i) (trainingLabels ! i)

-- @property
-- lipschitzRobust : Vector Bool n
-- lipschitzRobust = foreach i . 
--     lipschitzRobustSafetyAround (trainingInputs ! i) (trainingLabels ! i)

-- @property
-- robustEuclidean : Vector Bool n
-- robustEuclidean = foreach i . 
--     robustSafetyAroundEuclidean (trainingInputs ! i) (trainingLabels ! i)








