types_cols = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
              'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']

jam_cols = ['linqmap_subtype_JAM_HEAVY_TRAFFIC',
            'linqmap_subtype_JAM_MODERATE_TRAFFIC',
            'linqmap_subtype_JAM_STAND_STILL_TRAFFIC', ]

accident_cols = ['linqmap_subtype_ACCIDENT_MAJOR', 'linqmap_subtype_ACCIDENT_MINOR', ]
hazard_cols = ['linqmap_subtype_HAZARD_ON_ROAD',
               'linqmap_subtype_HAZARD_ON_ROAD_CAR_STOPPED',
               'linqmap_subtype_HAZARD_ON_ROAD_CONSTRUCTION',
               'linqmap_subtype_HAZARD_ON_ROAD_OBJECT',
               'linqmap_subtype_HAZARD_ON_ROAD_POT_HOLE',
               'linqmap_subtype_HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
               'linqmap_subtype_HAZARD_ON_SHOULDER_CAR_STOPPED', ]
road_closed_cols = ['linqmap_subtype_ROAD_CLOSED_CONSTRUCTION',
                    'linqmap_subtype_ROAD_CLOSED_EVENT']

subtypes_cols = jam_cols + accident_cols + hazard_cols + road_closed_cols
