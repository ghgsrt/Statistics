import { readFileSync } from 'fs';

function sum(arr: number[], cb?: (x: number, i: number) => number): number;
function sum<T extends any>(
	arr: T[],
	cb: (args: T, i: number) => number
): number;
function sum(arr: any[], cb: any) {
	return arr.reduce((acc, curr, i) => acc + (cb ? cb(curr, i) : curr), 0);
}

var Z_MAX = 6;
function poz(z: number) {
	var y, x, w;

	if (z == 0.0) {
		x = 0.0;
	} else {
		y = 0.5 * Math.abs(z);
		if (y > Z_MAX * 0.5) {
			x = 1.0;
		} else if (y < 1.0) {
			w = y * y;
			x =
				((((((((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w -
					0.019198292004) *
					w +
					0.059054035642) *
					w -
					0.151968751364) *
					w +
					0.319152932694) *
					w -
					0.5319230073) *
					w +
					0.797884560593) *
				y *
				2.0;
		} else {
			y -= 2.0;
			x =
				(((((((((((((-0.000045255659 * y + 0.00015252929) * y -
					0.000019538132) *
					y -
					0.000676904986) *
					y +
					0.001390604284) *
					y -
					0.00079462082) *
					y -
					0.002034254874) *
					y +
					0.006549791214) *
					y -
					0.010557625006) *
					y +
					0.011630447319) *
					y -
					0.009279453341) *
					y +
					0.005353579108) *
					y -
					0.002141268741) *
					y +
					0.000535310849) *
					y +
				0.999936657524;
		}
	}
	return z > 0.0 ? (x + 1.0) * 0.5 : (1.0 - x) * 0.5;
}

/*  CRITZ  --  Compute critical normal z value to
               produce given p.  We just do a bisection
               search for a value within CHI_EPSILON,
               relying on the monotonicity of pochisq().  */

function getCritZ(p: number) {
	var Z_EPSILON = 0.000001; /* Accuracy of z approximation */
	var minz = -Z_MAX;
	var maxz = Z_MAX;
	var zval = 0.0;
	var pval;
	if (p < 0.0) p = 0.0;
	if (p > 1.0) p = 1.0;

	while (maxz - minz > Z_EPSILON) {
		pval = poz(zval);
		if (pval > p) {
			maxz = zval;
		} else {
			minz = zval;
		}
		zval = (maxz + minz) * 0.5;
	}
	return zval;
}

function getPValue(z: number, tail: 'left' | 'right' | 'two') {
	if (tail === 'left') return poz(z);
	if (tail === 'right') return 1 - poz(z);
	return 2 * (1 - poz(Math.abs(z)));
}

function median(sample: number[]) {
	const sorted = sample.toSorted((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);

	if (mid % 2 === 0) return (sorted[mid] + sorted[mid + 1]) / 2;
	return sorted[mid];
}

function mean(sample: number[]) {
	return sum(sample) / sample.length;
}

function sumOfSquaredDeviations(sample: number[], sampleMean?: number) {
	sampleMean ??= mean(sample);

	return sum(sample, (x) => (x - sampleMean) ** 2);
	// return sample.reduce((acc, curr) => acc + (curr - sampleMean) ** 2, 0);
}

function sampleVariance(sample: number[]) {
	return sumOfSquaredDeviations(sample) / (sample.length - 1);
}
function populationVariance(population: number[]) {
	return sumOfSquaredDeviations(population) / population.length;
}
function sampleStandardDeviation(sample: number | number[]) {
	if (Array.isArray(sample)) return Math.sqrt(sampleVariance(sample));
	return Math.sqrt(sample);
}
function populationStandardDeviation(population: number | number[]) {
	if (Array.isArray(population))
		return Math.sqrt(populationVariance(population));
	return Math.sqrt(population);
}

function confidenceInterval(
	statistic: number,
	criticalValue: number,
	standardErrorOfStatistic: number
) {
	const marginOfError = criticalValue * standardErrorOfStatistic;
	return [statistic - marginOfError, statistic + marginOfError] as const;
}

function standardErrorDifferenceOfMeans(
	standardDeviation: [number, number],
	sampleSize: [number, number]
) {
	return Math.sqrt(
		standardDeviation[0] ** 2 / sampleSize[0] +
			standardDeviation[1] ** 2 / sampleSize[1]
	);
}

function regressionB1(
	points: [number, number][],
	xMean?: number,
	yMean?: number
) {
	const xSet = points.map(([x, _y]) => x);
	xMean ??= mean(xSet);
	yMean ??= mean(points.map(([_x, y]) => y));

	const numerator = sum(points, ([x, y]) => (x - xMean) * (y - yMean));
	// const numerator = points.reduce(
	// 	(total, [x, y]) => total + (x - xMean) * (y - yMean),
	// 	0
	// );
	const denominator = sumOfSquaredDeviations(xSet, xMean);

	return numerator / denominator;
}
function regressionB0(b1: number, xMean: number, yMean: number) {
	return yMean - b1 * xMean;
}

function regressionPredictedValues(
	points: [number, number][],
	b1?: number,
	b0?: number,
	xMean?: number,
	yMean?: number
) {
	xMean ??= mean(points.map(([x, _y]) => x));
	yMean ??= mean(points.map(([_x, y]) => y));

	b1 ??= regressionB1(points, xMean, yMean);
	b0 ??= regressionB0(b1Hat, xMean, yMean);

	return points.map(([x, _y]) => b0 + b1 * x);
}

function regressionResiduals(
	points: [number, number][],
	predictedValuesHat?: number[]
) {
	predictedValuesHat ??= regressionPredictedValues(points);

	return points.map(([_x, y], i) => y - predictedValuesHat[i]);
}

function regressionVarianceOfError(residuals: number[]) {
	const df = residuals.length - 2; //? minus 2 accounts for the two parameters (b0 and b1) estimated from the data
	return sum(residuals, (residual) => residual ** 2) / df;
}

function regressionVarianceOfB1(
	points: [number, number][],
	xMean?: number,
	varianceOfError?: number
) {
	const xValues = points.map(([x, _y]) => x);

	xMean ??= mean(xValues);
	varianceOfError ??= regressionVarianceOfError(regressionResiduals(points));

	return varianceOfError / sumOfSquaredDeviations(xValues, xMean);
}
function regressionVarianceOfB0(
	points: [number, number][],
	xMean?: number,
	varianceOfError?: number
) {
	const xValues = points.map(([x, _y]) => x);
	const n = xValues.length;

	xMean ??= mean(xValues);
	varianceOfError ??= regressionVarianceOfError(regressionResiduals(points));

	return (
		varianceOfError /
		(1 / n + xMean ** 2 / sumOfSquaredDeviations(xValues, xMean))
	);
}

function regressionStandardErrorOfB1(
	points: [number, number][],
	xMean?: number,
	varianceOfError?: number
) {
	return Math.sqrt(regressionVarianceOfB1(points, xMean, varianceOfError));
}
function regressionStandardErrorOfB0(
	points: [number, number][],
	xMean?: number,
	varianceOfError?: number
) {
	return Math.sqrt(regressionVarianceOfB0(points, xMean, varianceOfError));
}

function summaryStatistics(data: number[], type: 'sample' | 'population') {
	const _data = data.toSorted((a, b) => a - b);

	const _median = median(_data);
	const _mean = mean(_data);
	const _standardDeviation =
		type === 'sample'
			? sampleStandardDeviation(_data)
			: populationStandardDeviation(_data);
	const _range = [_data[0], _data[_data.length - 1]];

	return [_median, _mean, _standardDeviation, _range] as const;
}

//! --------------------------------------------------------------------------------------------------------------------------
//! Initialize Data
//! --------------------------------------------------------------------------------------------------------------------------

const normalize = ['Low', 'Moderate', 'High'];
const data = readFileSync(
	'./students_adaptability_level_online_education.csv',
	{ encoding: 'utf8', flag: 'r' }
);
const rows: [number, number][] = data.split('\r\n').map((row) => {
	const columns = row.split(',');
	return [
		columns[0] === 'Boy' ? 0 : 1,
		normalize.indexOf(columns[columns.length - 1]),
	];
});

let boys: number[] = [];
let girls: number[] = [];
for (const [gender, adaptivity] of rows) {
	if (gender === 0) boys.push(adaptivity);
	else girls.push(adaptivity);
}

//* NOTES BEFORE WE BEGIN:::
//* 	I'm tutoring mostly for math. I can help with ensuring you've done the appropriate steps and achieved
//*			your desired outcomes with the numbers, but I will be of very little help when it comes to the
//*			actual interpretations of the results

//* Where relevent:
//* 	gender is normalized to 0 for boys, 1 for girls
//*		adaptability is normalized to 0 for low, 1 for moderate, 2 for high

//! --------------------------------------------------------------------------------------------------------------------------
//! Summary Statistics
//! --------------------------------------------------------------------------------------------------------------------------

let totalDataset = [...boys, ...girls];
let [totalMedian, totalMean, totalStandardDeviation, totalRange] =
	summaryStatistics(totalDataset, 'sample');

let [boysMedian, boysMean, boysStandardDeviation, boysRange] =
	summaryStatistics(boys, 'sample');
let [girlsMedian, girlsMean, girlsStandardDeviation, girlsRange] =
	summaryStatistics(girls, 'sample');

//? Should probably ensure you're including these along with the frequencies
//? I'd also HIGHLY recommend you add graphs for a lot of this stuff
//*	Some ideas:
//*		- Boxplot of adaptability levels by gender
//*		- Histogram (or density plot) of adaptability levels
//*		- Bar graph of the mean adaptability levels (straightforward visualization of the two-sample test below, I think)

//! --------------------------------------------------------------------------------------------------------------------------
//! Two-Sample Z-test
//! --------------------------------------------------------------------------------------------------------------------------

//? We have 2 samples
//?		per sample, n is much greater than 30, therefore we can treat the respective sample standard
//?		deviations as population, and we may assume a normal distribution
//? Therefore, we use a two-sample z-test for our test statistic

//! NOTE: test statistic always === (sampleStatistic - nullHypothesisValue) / SE(sampleStatistic)
//! this will also apply in the regression analysis section when we get to hypothesis testing

//? Let's begin with hypothesis testing using the means as our sample statistic
//? 	Specifically, the difference of means as we're comparing the means of two samples
//? null hypothesis: boysMean = girlsMean   (i.e., mean adaptability of boys is equal to mean adaptability of girls)
//? alt  hypothesis: boysMean =/= girlsMean (i.e., adaptability of boys is NOT equal to mean adaptability of girls)
//? Therefore, two-tail when determining p-value

//* NOTE: that our null hypothesis can be rewritten as:
//*		boysMean - girlsMean = 0
//* This is how we determined that:
//* 	1. our sample statistic is the difference of means (lhs of hypotheses is the sample statistic)
//*		2. the value of our null hypothesis is 0, allowing us to simplify the test statistic formula to:
//*			sampleStatistic / SE(sampleStatistic)

//? First, we need the sample statistic (see above)
let differenceOfMeans = boysMean - girlsMean;

//? Next, the standard error of the sample statistic
//? Standard error is always the standard deviation of the sample statistic divided by the square root of n,
//?		unless you're dealing with two samples, in which case it's a little different
//? 	Rather than being the sqrt(variance / n) (this is equivalent to what I said above),
//?		you take the variance / n for each sample, add them together, and root the whole thing (see formula sheet)
let standardError = standardErrorDifferenceOfMeans(
	[boysStandardDeviation, girlsStandardDeviation],
	[boys.length, girls.length]
);

//? Now we can compute the test statistic:
let z = differenceOfMeans / standardError;

//? Then the p-value using the normal distribution
let pValue = getPValue(z, 'two');

//? Let's assume a significance of 0.05
let alpha = 0.05;

//? If p-value is less than alpha, reject the null hypothesis
let reject = pValue < alpha;

//? Two-sample confidence interval ------------------------------------------------------------------------------------------

//? need the critical value of z for alpha/2
let critZ = getCritZ(alpha / 2);

let [twoSampleCI_low, twoSampleCI_high] = confidenceInterval(
	differenceOfMeans,
	critZ,
	standardError
);

//! --------------------------------------------------------------------------------------------------------------------------
//! Regression Analysis
//! --------------------------------------------------------------------------------------------------------------------------

//? Regression Analysis:
//? 	dependent   variable: adaptability (Y) (we're claiming this "depends" on gender)
//? 	independent variable: gender	   (X) (we're claiming this does not depend on anything)
//? the 'rows' array from above is currently in the shape [gender, adaptability] for each row, which matches
//? 	what we want for the set of points (x, y)
//* NOTE: that I normalized the values for gender and adaptability such that:
//* 	gender is actually 0 for boys and 1 for girls
//*		adaptability is actually 0 for low, 1 for moderate, and 2 for high
let points = rows; // e.g., [[0, 1], [0,2], [1,1], [1,0], etc.]

//? first, we want to calculate the slope (b1)
//? this relies on the means of X and Y:
let xValues = points.map(([x, _y]) => x); // the set of all gender values [0, 0, 1, 0, 1, 1, etc.]
let yValues = points.map(([_x, y]) => y); // the set of all adaptability values [0, 2, 1, 2, 0, 1, etc.]

let meanX = mean(xValues);
let meanY = mean(yValues);

let b1Hat = regressionB1(points, meanX, meanY); // hat just means it's an estimate

//? next, the intercept (b0), which relies on b1
let b0Hat = regressionB0(b1Hat, meanX, meanY);

//? Now, we can predict the values for Y at any given X
//? 	Even though gender is categorical (can only be either 0 or 1), we still want to calculate the
//? 		predicted values for every single point in the dataset rather than just a single time each for 0 and 1
//?		This is because later calculations rely on the number of predicted values matching the original
//?			sample
let predictedValuesHat = regressionPredictedValues(
	points,
	b1Hat,
	b0Hat,
	meanX,
	meanY
);

//? Hypothesis Testing -------------------------------------------------------------------------------------------------------

//? Now that all of the prep is done, we can move on to the hypothesis testing and confidence intervals
//? First, the hypothesis testing (ensure it correlates with the hypotheses from two-sample testing)
//?		Let's use b1 for our sample statistic as it should tell us the correlation b/w gender and adaptability:
//*			(NOTE: b0 is basically meaningless for our scenario)
//? 	null hypothesis: b1 = 0   (i.e., gender does NOT affect adaptability level)
//? 	alt  hypothesis: b1 =/= 0 (i.e., gender does affect adaptability level)
//? Therefore, two-tailed when determing p-value, as with the two-sample testing

//? We already calculated our sample statistic (b1) above, so now we just need to get its standard error
//? To do so, we first need to calculate the variance of the residuals, found using the set of
//? predicted values we computed earlier

//? First, we need the actual residuals:
//? (epsilon in the regression equation; difference between the observed and predicted adaptability levels)
let residuals = regressionResiduals(points, predictedValuesHat);

//? Next, we calculate the residual variance
//? (NOTE: the degrees of freedom = n - 2, where the minus 2 accounts for b1 & b0 being estimated from the data)
let residualVariance = regressionVarianceOfError(residuals);

//? Finally, we can get the standard error for b1:
let standardErrorB1Hat = regressionStandardErrorOfB1(
	points,
	meanX,
	residualVariance
);

//? While we can use z-score with a normal distribution, regression analysis traditionally uses
//? 	the t-distribution, which is what I reccomend you use. For our purposes here I'll be using z,
//?		but at the scale of our dataset the t-distribution should converge on a normal distribution,
//?		so not only will the calculations be identical, but our numbers also should be almost identical.
//?		Just remember to assume anywhere you see me using functions relating to 'z' you should actually be
//?		doing it in the context of the t-distribution

//? First, let's calculate the test statistic using b1 as the sample statistic
let t = b1Hat / standardErrorB1Hat;

//? Next, the p-value (two-tail)
pValue = getPValue(t, 'two');

//? let's once again assume a significance of 0.05
alpha = 0.05;

//? If p-value is less than alpha, reject the null hypothesis
reject = pValue < alpha;

//? Confidence Interval ------------------------------------------------------------------------------------------------------

//? Calculate the degrees of freedom. Remember from earlier this is n - 2
let df = xValues.length - 2; // n is the number of rows in the dataset

//? Use the t-table or whatever to find the t-value corresponding to the df and alpha
//? Although, assuming our dataset is large enough, you should get a value pretty close to the 1.96
//? that I'll be getting here using the normal distribution
let tCrit = getCritZ(alpha);

//? Now we just compute the confidence intervals as usual:
let [regressionCI_low, regressionCI_high] = confidenceInterval(
	b1Hat,
	tCrit,
	standardErrorB1Hat
);
