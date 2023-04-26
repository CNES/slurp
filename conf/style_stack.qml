<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" minScale="1e+8" version="3.10.1-A Coruña" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
  </customproperties>
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry alpha="255" label="inconnu" value="0" color="#ffffff"/>
        <paletteEntry alpha="255" label="Veg basse" value="1" color="#18df2f"/>
        <paletteEntry alpha="255" label="Veg haute" value="2" color="#0f7637"/>
        <paletteEntry alpha="255" label="Eau" value="3" color="#2895cb"/>
        <paletteEntry alpha="255" label="Urbain" value="4" color="#ff0184"/>
        <paletteEntry alpha="255" label="Route / parking ?" value="7" color="#ffb8dd"/>
        <paletteEntry alpha="255" label="Eau (prédiction)" value="8" color="#2bc7f7"/>
        <paletteEntry alpha="255" label="Ombre" value="9" color="#e2ff01"/>
        <paletteEntry alpha="255" label="Urbain (faux positifs)" value="10" color="#969db0"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors"/>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0" colorizeRed="255" colorizeGreen="128" colorizeStrength="100" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
