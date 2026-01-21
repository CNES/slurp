# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is blocked to the next chosen version and name of the milestone at a given date.
A new section Unreleased is opened then for next dev phase.

## Unreleased (shall be 0.0.6 - 2026-01-22)

### Added
	- Analyse vegetation clustering : there are new options to fit vegetation clustering with what is observed in a global land cover map : slurp_vegetationmask -autolabel [-labeling_strategy nearest/overestimate/underestimate]. Note that the "pct_veg/low_veg/etc." are pourcentage of observed class in the global land cover map (see slurp_prepare --analyse_glcm). See #24
	- Analysis of global world cover map is activated by default (and OTB based script had been updated). See #23 / #14
	- Add possibility to use a nearly void config files : almost all config files shall be initiated with default values #14

### Changed
	- Change convention for validity mask : 0 values shall stand for "valid" pixels, and other values shall stand for user specified reasons (ex : 1 for NO_DATA in input VHR image, 2 for clouds, etc.) --> see #32

### Fixed
	- fix precommit (#26)
	- fix sphinx doc tests (#25)
	- improve water bodies categorization to fix #33 : the idea is to analyse all categorized water detections and merge neighboured areas that have been separated by tiling.

## 0.0.4 First Official Release (2025-09-25)

	First version on Pypi
### Added
