const float kSheenPi = 3.14159265359;
const float kSheenEpsilon = 1.0e-6;

float sheenAlpha(float sheenRoughness) {
  sheenRoughness = max(sheenRoughness, kSheenEpsilon);
  return sheenRoughness * sheenRoughness;
}

float DCharlie(float sheenRoughness, float ndoth) {
  float alphaG = sheenAlpha(sheenRoughness);
  float invR = 1.0 / alphaG;
  float cos2h = ndoth * ndoth;
  float sin2h = max(1.0 - cos2h, 0.0);
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * kSheenPi);
}

float lambdaSheenNumericHelper(float x, float alphaG) {
  float oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
  float a = mix(21.5473, 25.3245, oneMinusAlphaSq);
  float b = mix(3.82987, 3.32435, oneMinusAlphaSq);
  float c = mix(0.19823, 0.16801, oneMinusAlphaSq);
  float d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
  float e = mix(-4.32054, -4.85967, oneMinusAlphaSq);
  return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float lambdaSheen(float cosTheta, float alphaG) {
  float safeCosTheta = clamp(abs(cosTheta), 0.0, 1.0);
  if (safeCosTheta < 0.5) {
    return exp(lambdaSheenNumericHelper(safeCosTheta, alphaG));
  }

  return exp(2.0 * lambdaSheenNumericHelper(0.5, alphaG) -
             lambdaSheenNumericHelper(1.0 - safeCosTheta, alphaG));
}

float VSheen(float ndotl, float ndotv, float sheenRoughness) {
  float alphaG = sheenAlpha(sheenRoughness);
  ndotv = max(ndotv, kSheenEpsilon);
  ndotl = max(ndotl, kSheenEpsilon);
  return clamp(1.0 / ((1.0 + lambdaSheen(ndotv, alphaG) +
                       lambdaSheen(ndotl, alphaG)) *
                      (4.0 * ndotv * ndotl)),
               0.0, 1.0);
}

float albedoSheenScalingFactor(float ndotv, float sheenRoughness) {
  ndotv = clamp(ndotv, kSheenEpsilon, 1.0);
  float c = 1.0 - ndotv;
  float c3 = c * c * c;
  return 0.65584461 * c3 +
         1.0 / (4.16526551 +
                exp(-7.97291361 * sqrt(sheenRoughness) + 6.33516894));
}
