#include "common.h"


// Vertex attributes.
struct Interpolants {
  vec3  pos;     // World-space vertex position
  vec3  normal;  // World-space vertex normal
  vec4  color;   // Linear-space color
  float depth;   // Z coordinate after applying the view matrix (larger = further away)
};

[[using spirv: in, location(0)]]
Interpolants IN;

[[using spirv: uniform, binding(UBO_SCENE)]]
SceneData scene;

// Gooch shading!
// Interpolates between white and a cooler color based on the angle
// between the normal and the light.
inline vec3 goochLighting(vec3 normal) {
  // Light direction
  vec3 light = normalize(vec3(-1, 2, 1));
  // cos(theta), remapped into [0,1]
  float warmth = dot(normalize(normal), light) * 0.5f + 0.5f;
  // Interpolate between warm and cool colors (alpha here will be ignored)
  return mix(vec3(0, 0.25, 0.75), vec3(1, 1, 1), warmth);
}

// Applies Gooch shading to a surface with color and alpha and returns
// an unpremultiplied RGBA color.
inline vec4 shading(Interpolants its, const SceneData& scene) {
  vec3 colorRGB = its.color.rgb * goochLighting(its.normal);

  // Calculate transparency in [alphaMin, alphaMin+alphaWidth]
  float alpha = clamp(scene.alphaMin + its.color.a * scene.alphaWidth, 0.f, 1.f);

  return vec4(colorRGB, alpha);
}

// Converts an unpremultiplied scalar from linear space to sRGB. Note that
// this does not match the standard behavior outside [0,1].
inline float unPremultLinearToSRGB(float c) {
  return c < 0.0031308f ?
    c * 12.92f :
    pow(c, 1.0f / 2.4f) * 1.055f - 0.055f;
}

// Converts an unpremultiplied RGB color from linear space to sRGB. Note that
// this does not match the standard behavior outside [0,1].
inline vec4 unPremultLinearToSRGB(vec4 c) {
  c.r = unPremultLinearToSRGB(c.r);
  c.g = unPremultLinearToSRGB(c.g);
  c.b = unPremultLinearToSRGB(c.b);
  return c;
}


template<int OitLayer>
[[spirv::vert]]
void oit_simple_vert() {
  vec4 color = shading(IN, scene);

  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  // Get the number of pixels in the image.
  int viewSize = scene.viewport.z;

  // Get the index of the current sample at the current fragemnet.
  int listPos = viewSize * OitLayer * glfrag_SampleID + 
    glfrag_FragCoord.y * scene.viewport.x + glfrag_FragCoord.x;

  uvec4 storeValue(
    packUnorm4x8(sRGBColor), 
    floatBitsToUint(glfrag_FragCoord.z), 
    storeMask, 
    0
  );

}