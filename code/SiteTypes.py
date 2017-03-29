from collections import namedtuple

# default parameter values for r, u & lambda
SiteTypes = namedtuple('SiteTypes', 'hom het nTypes typesList')
siteTypes = SiteTypes(0, 1, 2, [0,1,2])
