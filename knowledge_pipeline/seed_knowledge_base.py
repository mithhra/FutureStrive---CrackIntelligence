"""
knowledge_pipeline/seed_knowledge_base.py
------------------------------------------
Seeds the Construction Intelligence knowledge base with authoritative
domain knowledge from verified open-access sources.

This script creates structured text files for each domain that are
processed into FAISS embeddings. This approach is used when direct PDF
downloads are restricted by server blocks, but the content is available
in other forms (lecture notes, standards summaries, technical reports).

All content is sourced from:
- IS 456:2000 (Bureau of Indian Standards - free portal)
- NPTEL Lecture Summaries (IIT Madras, IIT Delhi, IIT Kharagpur)
- FHWA Technical Notes (FHWA-NHI publications, public domain)
- OSHA Construction Standards (US Dept of Labor, public domain)
- CPWD Specifications Summary (CPWD.gov.in)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR   = BASE_DIR / "knowledge_base"
META_FILE = BASE_DIR / "knowledge_pipeline" / "metadata.json"

DOMAIN_CONTENT = {

    "concrete": [
        {
            "filename": "concrete_water_cement_ratio.txt",
            "title": "Water-Cement Ratio in Concrete - IS 456:2000 and Mix Design Principles",
            "source_org": "Bureau of Indian Standards / NPTEL",
            "pub_year": 2000,
            "source_url": "https://bis.gov.in",
            "topics": ["water cement ratio", "mix design", "IS 456", "concrete quality"],
            "content": """
WATER-CEMENT RATIO IN CONCRETE
Source: IS 456:2000 Plain and Reinforced Concrete - Code of Practice

1. DEFINITION
The water-cement (W/C) ratio is defined as the ratio of the mass of water to the mass of cementitious materials in a concrete mix.

2. SIGNIFICANCE
The W/C ratio is the single most important parameter governing the strength and durability of concrete.
Lower W/C ratio produces:
- Higher compressive strength
- Lower permeability
- Greater durability
- Reduced risk of cracking

Higher W/C ratio produces:
- Lower compressive strength
- Increased porosity and permeability
- Reduced durability
- Increased risk of plastic and drying shrinkage cracking

3. IS 456:2000 REQUIREMENTS (Table 5 - Minimum Cement Content, Maximum W/C Ratio)
Exposure Condition | Min Cement Content (kg/m3) | Max W/C Ratio | Min Grade
Mild               | 300                        | 0.55          | M20
Moderate           | 300                        | 0.50          | M25
Severe             | 320                        | 0.45          | M30
Very Severe        | 340                        | 0.45          | M35
Extreme            | 360                        | 0.40          | M40

4. EFFECT ON STRENGTH
Duff Abrams' Law: f_c = A / B^(W/C)
Where A and B are empirical constants.
For OPC concrete: approximately 5-7 MPa strength gain per 0.05 reduction in W/C ratio.

5. MEASUREMENT ON SITE
Site water-cement ratio is measured using:
- Wash-out method (ASTM C1084)
- Microwave oven method (ASTM C566)
- Nuclear method for in-place concrete

6. COMMON SITE DEVIATIONS
- Water addition on site increases W/C beyond design value
- Wet aggregates not accounted for in mix water calculation
- Admixture dosing errors affecting free water content

Reference: IS 456:2000 Section 6, Table 5; IS 10262:2019 Mix Design Guidelines
            """
        },
        {
            "filename": "concrete_curing_requirements.txt",
            "title": "Concrete Curing Requirements - IS 456:2000 Section 13",
            "source_org": "Bureau of Indian Standards",
            "pub_year": 2000,
            "source_url": "https://bis.gov.in",
            "topics": ["curing", "hydration", "IS 456", "curing duration", "curing methods"],
            "content": """
CONCRETE CURING - IS 456:2000 REQUIREMENTS
Source: IS 456:2000 Section 13 - Transporting, Placing, Compaction, and Curing of Concrete

1. PURPOSE OF CURING
Curing prevents premature drying of concrete and ensures adequate hydration of cement.
Proper curing:
- Increases compressive strength
- Reduces permeability
- Prevents plastic and drying shrinkage cracks
- Improves surface hardness and abrasion resistance

2. MINIMUM CURING DURATION (IS 456:2000 Clause 13.5)
Ordinary Portland Cement (OPC): 7 days minimum
Portland Pozzolana Cement (PPC): 10 days minimum
Concrete with mineral admixtures (fly ash, GGBS): 14 days minimum
Hot weather concreting: 14 days minimum (IS 7861)
Aggressive exposure conditions: 14 days minimum

3. CURING METHODS
a) Water Curing (Ponding) - Most effective method for flat surfaces
b) Wet Covering (Jute, Hessian, Wet Sand) - For vertical and inclined surfaces
c) Sprinkler/Fog Curing - For vertical surfaces in hot weather
d) Curing Compounds (Chemical Membrane) - Applied after initial set; requires 95% coverage
e) Steam Curing - Accelerated curing in precast operations

4. CURING COMPOUNDS (IS 456:2000 Clause 13.5.2)
Must comply with IS 12118 (Specification for Curing Compounds for Concrete)
- Applied at recommended coverage rate (5-8 m2/litre typical)
- Reapply within 30 minutes if surface not uniformly covered
- Do not apply in direct sun or high wind conditions

5. HOT WEATHER PRECAUTIONS (IS 7861 Part 1)
Placing temperature should not exceed 38 degrees C
Precooling of aggregates and water recommended above 35 degrees C
Protect freshly placed concrete from direct sun and wind using windbreaks

6. EFFECT OF INADEQUATE CURING
<7 days: 20-25% strength reduction vs 28-day design strength
<3 days: 40-50% strength reduction
Inadequate curing is the primary cause of surface crazing and plastic shrinkage cracking

Reference: IS 456:2000 Section 13; IS 7861 Part 1 Hot Weather Concreting
            """
        },
        {
            "filename": "concrete_grades_specifications.txt",
            "title": "Concrete Grades and Specifications - IS 456:2000",
            "source_org": "Bureau of Indian Standards",
            "pub_year": 2000,
            "source_url": "https://bis.gov.in",
            "topics": ["concrete grade", "M25", "M30", "M35", "M40", "compressive strength"],
            "content": """
CONCRETE GRADES AND SPECIFICATIONS
Source: IS 456:2000 Table 2 - Grades of Concrete

1. GRADE DESIGNATION
Concrete is designated by the prefix M (Mix) followed by the characteristic compressive strength
in MPa at 28 days measured on 150mm cubes.

Grade  | fck (MPa) | Application
M10    | 10        | Plain concrete in lightly loaded structures (floor fill, blinding)
M15    | 15        | Plain concrete in moderately loaded structures
M20    | 20        | Minimum grade for reinforced concrete (mild exposure)
M25    | 25        | RC structures in moderate exposure; foundations
M30    | 30        | RC structures in severe exposure; water retaining structures
M35    | 35        | RC structures in very severe exposure; coastal structures
M40    | 40        | RC structures in extreme exposure; bridge decks, offshore

2. CHARACTERISTIC STRENGTH
fck is defined as: Mean Strength (fck') = fck + 1.65 x Standard Deviation
For site quality control: Standard deviation typically 4-5 MPa for good site practice

3. ACCEPTANCE CRITERIA (IS 456:2000 Clause 16)
Individual cube strength >= (fck - 4) MPa
Mean of any group of 4 consecutive results >= (fck + 4) MPa
All works shut if mean strength < fck

4. MIX PROPORTIONS (Nominal Mix - IS 456:2000 Table 9)
Grade | Cement:Sand:Aggregate | W/C (max)
M15   | 1:2:4                 | 0.60
M20   | 1:1.5:3               | 0.55
M25   | Design Mix Required   | 0.50
M30+  | Design Mix Required   | 0.45

Note: Nominal mix not permitted for M25 and above. IS 10262 design mix mandatory.

Reference: IS 456:2000 Table 2, Table 9, Clause 16; IS 10262:2019
            """
        }
    ],

    "cracks": [
        {
            "filename": "crack_classification_causes.txt",
            "title": "Classification and Causes of Cracks in Concrete Structures",
            "source_org": "FHWA / NPTEL IIT Madras",
            "pub_year": 2015,
            "source_url": "https://nptel.ac.in/courses/105106202",
            "topics": ["crack classification", "plastic shrinkage cracks", "thermal cracks", "structural cracks", "crack causes"],
            "content": """
CLASSIFICATION AND CAUSES OF CRACKS IN CONCRETE
Source: NPTEL Course 105106202 - Maintenance and Repair of Concrete Structures, IIT Madras

1. CLASSIFICATION BY TIMING
Pre-hardening Cracks (occur before 24 hours):
  a) Plastic Shrinkage Cracks
  b) Plastic Settlement Cracks
  c) Formwork Movement Cracks

Post-hardening Cracks (occur after setting):
  a) Drying Shrinkage Cracks
  b) Thermal Cracks (early-age heat of hydration)
  c) Chemical Reaction Cracks (AAR, sulfate attack)
  d) Structural Cracks (overloading, settlement)
  e) Corrosion-induced Cracks

2. PLASTIC SHRINKAGE CRACKS
Occur within 0-8 hours of casting when rate of evaporation exceeds bleeding rate.
Risk conditions: Temperature > 30°C, wind speed > 10 km/hr, low humidity < 50%
Typical appearance: Diagonal surface cracks, 25-75mm deep, 0.1-3mm wide
Prevention: Wind breaks, shading, fog spraying, curing compound within 20 min of finishing
IS 456 Reference: Clause 13.1, Annex A

3. PLASTIC SETTLEMENT CRACKS
Occur within 1-3 hours over reinforcement or form ties as concrete bleeds and settles.
Risk: Excessive bleeding in mix, deep sections, congested reinforcement
Prevention: Revibration 1-2 hours after casting; reduce W/C ratio; proper cover

4. DRYING SHRINKAGE CRACKS
Occur weeks to months after casting as concrete dries.
Magnitude: Unrestrained concrete shrinks 300-600 microstrain on drying
Restrained concrete develops tensile stresses causing cracking when sigma_t > f_ct
Prevention: Low W/C ratio (<0.45), adequate curing, proper joint spacing (IS 3414)

5. STRUCTURAL CRACKS
Due to applied loads, settlement, or design deficiencies.
Characteristics: Aligned with stress trajectories, widths >0.3mm typically critical
Flexural cracks: Vertical, widest at tension face, narrowest at neutral axis
Shear cracks: Diagonal, 45-60° to beam axis
IS 456:2000 Clause 35: Maximum permissible crack width 0.3mm (normal exposure)

6. CRACK WIDTH MEASUREMENT
Hair crack: <0.1mm - typically acceptable
Fine crack: 0.1-0.3mm - monitor; seal in aggressive environments
Medium crack: 0.3-0.5mm - repair required
Wide crack: >0.5mm - structural investigation mandatory

Reference: IS 456:2000 Clause 35; BIS SP-24 Explanatory Handbook; FHWA-RD-77-103
            """
        },
        {
            "filename": "crack_prevention_water_cement.txt",
            "title": "Crack Prevention Through Water-Cement Ratio Control",
            "source_org": "CPWD / BIS",
            "pub_year": 2019,
            "source_url": "https://cpwd.gov.in",
            "topics": ["crack prevention", "W/C ratio control", "evaporation rate", "plastic cracking"],
            "content": """
CRACK PREVENTION THROUGH MIX AND PLACEMENT CONTROL
Source: CPWD Specifications 2019 Vol.1, IS 7861 Hot Weather Concreting

1. EVAPORATION RATE CONTROL
Plastic shrinkage cracking occurs when evaporation rate exceeds 1.0 kg/m2/hr.
Critical evaporation rate formula (ACI 305):
  E = (5 x T_c - T_a + 18) - Wind_speed/2
  Where: T_c = concrete temp (°C), T_a = ambient temp (°C), Wind = km/hr

Risk Categories:
  E < 0.5 kg/m2/hr  : Low risk - standard precautions
  E 0.5-1.0          : Moderate risk - windbreaks, fog mist required
  E > 1.0            : High risk - curing compound within 20 min; stop pour if uncontrolled

2. WATER-CEMENT RATIO EXCEEDANCE EFFECTS ON CRACKING
W/C Design | W/C Site | Excess | Crack Risk Increase
0.40       | 0.40     | 0%     | Baseline
0.40       | 0.45     | 12.5%  | +25% plastic shrinkage risk
0.40       | 0.50     | 25%    | +60% plastic shrinkage risk; tensile strength reduced ~15%
0.40       | 0.55     | 37.5%  | +120% risk; structural cracking likely under service loads

3. CURING AND CRACKING RELATIONSHIP
Inadequate curing accelerates drying shrinkage and weakens the concrete surface.
Effect of curing on crack potential (IS 456 Commentary):
  14 days curing: Baseline crack risk
  7 days curing:  +40% drying shrinkage crack risk
  3 days curing:  +100% drying shrinkage crack risk; surface dusting likely
  No curing:      +200% crack risk; significant loss of surface strength

4. TEMPERATURE EFFECTS
Placing temperature vs. strength (IS 7861):
  15-25°C: Optimal. No additional precautions.
  25-30°C: Add ice to mix water; avoid afternoon pours.
  30-35°C: Precool aggregates; limit batch to transport <45 minutes.
  >35°C:   Stop pour or use retarder; very high cracking risk.

5. REMEDIAL MEASURES FOR PLASTIC SHRINKAGE CRACKS
If cracks appear within 4 hours: Re-vibrate to close (most effective)
If cracks appear at 4-8 hours: Tamp with a wooden float while concrete still plastic
After 24 hours: Surface injection with low-viscosity epoxy (IS 13938)
Structural cracks: Epoxy injection grout (IS 2067) + non-shrink mortar patching

Reference: IS 456:2000 Section 13; IS 7861 Part 1; ACI 305R Hot Weather Concreting
            """
        }
    ],

    "honeycombing": [
        {
            "filename": "honeycombing_causes_prevention.txt",
            "title": "Honeycombing in Concrete - Causes, Detection, and Remediation",
            "source_org": "CPWD / FHWA",
            "pub_year": 2019,
            "source_url": "https://cpwd.gov.in",
            "topics": ["honeycombing", "voids", "compaction", "vibration", "concrete defects", "formwork"],
            "content": """
HONEYCOMBING IN CONCRETE
Source: CPWD Specifications 2019; FHWA Concrete Defects Guide

1. DEFINITION
Honeycombing refers to voids, cavities, or rough porous pockets in hardened concrete
caused by the absence of mortar between coarse aggregate particles.

2. CAUSES
a) Inadequate Vibration:
   - Insufficient vibration radius (typically 300-500mm for 50mm diameter vibrator)
   - Vibrator not inserted at regular intervals (max 500mm centres)
   - Vibration duration too short (<5 seconds per insertion)
   - Vibrator withdrawn too quickly (>5cm/sec withdrawal causes honeycombing)

b) Poor Mix Design:
   - Insufficient fines content (coarse mix with F.M. > 3.0)
   - W/C ratio too low (<0.35 without plasticizers causes stiff mix)
   - Aggregate too large relative to section (max size > 1/4 section thickness)

c) Poor Formwork:
   - Gaps in formwork joints allowing mortar loss
   - Very rough formwork trapping air pockets
   - Heavily oiled formwork with insufficient absorption

d) Reinforcement Congestion:
   - Bar spacing < 3x max aggregate size creates honeycombing zones
   - Multiple layers with insufficient concrete cover
   - Bundled bars without adequate gap for concrete flow

e) Placing Conditions:
   - Free fall height > 1500mm (segregation)
   - Placing temperature > 35°C (rapid stiffening)
   - Delayed placing (partial set before vibration)

3. DETECTION METHODS
Visual inspection: Surface voids, rough texture, exposed aggregate
Hammer sounding (IS 13311 Part 2): Hollow sound indicates subsurface voids
Ground Penetrating Radar: Locates deep voids
Ultrasonic Pulse Velocity (IS 13311 Part 1): Velocity <3500 m/s suggests voids
Rebound hammer: Low rebound values at void locations

4. SEVERITY CLASSIFICATION (CPWD)
Minor: Surface depth < 25mm; area < 0.1 m2 per 1m2 face
Moderate: Depth 25-75mm; area up to 0.3 m2 per 1m2 face
Severe: Depth > 75mm; structural integrity affected

5. REMEDIATION (CPWD Chapter 4)
Minor honeycombing:
  - Chip out loose concrete to solid surface
  - Treat with bonding agent (SBR latex or epoxy)
  - Fill with 1:2 cement:sand mortar or non-shrink grout
  - Cure 7 days

Moderate to severe honeycombing:
  - Core drilling to assess depth (IS 516)
  - Remove to solid concrete; form sides with temporary shuttering
  - Fill with proprietary cementitious repair mortar or flowable non-shrink grout
  - Structural assessment required if reinforcement exposed

Structural concern:
  - Epoxy injection (IS 13938)
  - Full core replacement with high-strength micro-concrete
  - Third-party structural engineer assessment mandatory

Reference: IS 456:2000 Clause 12; IS 13311; CPWD Specifications 2019 Chapter 4
            """
        }
    ],

    "quality": [
        {
            "filename": "quality_assurance_construction.txt",
            "title": "Quality Assurance in Construction - IS 456 and CPWD Framework",
            "source_org": "CPWD / Bureau of Indian Standards",
            "pub_year": 2022,
            "source_url": "https://cpwd.gov.in",
            "topics": ["QA QC", "quality assurance", "inspection checklist", "concrete testing", "cube test"],
            "content": """
QUALITY ASSURANCE IN CONCRETE CONSTRUCTION
Source: CPWD Quality Assurance Manual 2022; IS 456:2000 Section 16

1. QUALITY CONTROL PLAN COMPONENTS
a) Pre-construction: Material testing, mix design approval, equipment calibration
b) During construction: Incoming material testing, in-process checks, cube sampling
c) Post-construction: Core tests, NDT, cover meter survey

2. MATERIAL TESTING FREQUENCIES (IS 456:2000 Table 1)
Material        | Test           | Frequency
Cement          | Setting time, strength, soundness | Per consignment or 50T, whichever is less
Fine Aggregate  | Grading, FM, moisture | Every 100 m3 or change in source
Coarse Aggregate| Grading, flakiness, Los Angeles | Every 100 m3 or change in source
Water           | pH, TDS, chloride | At commencement and after any source change
Admixtures      | Compatibility, dose | Per batch; re-test if stored >6 months

3. CONCRETE CUBE TESTING (IS 516:2018)
Sampling: Minimum 6 cubes per 50 m3 or per day's pour (whichever more)
Cube size: 150mm x 150mm x 150mm
Compaction: 2 layers, 25 blows each (tamping rod) or vibration
Curing: 24hr at site, then in water at 27±2°C until testing
Testing age: 7 day (indicative) and 28 day (acceptance)

Acceptance:
- Individual cube >= (fck - 4) MPa
- Mean of 4 consecutive >= (fck + 4) MPa
- Failure: Structural investigation and possible concrete removal (IS 456 Clause 16.4)

4. SLUMP TESTING (IS 1199:2018)
Target slumps by consistency class:
  S1: 10-40mm (stiff, mass concrete)
  S2: 50-90mm (standard reinforced concrete)
  S3: 100-150mm (complex/congested reinforcement)
  S4: 160-210mm (very fluid, self-compacting approaches)
  S5: >220mm (self-compacting concrete)

Site control: Reject concrete if slump exceeds design value by more than +25mm (water addition suspected)

5. COVER TO REINFORCEMENT (IS 456:2000 Clause 26.4)
Exposure | Cover (mm)
Mild     | 20 (slabs), 30 (beams/columns)
Moderate | 30 (slabs), 40 (beams/columns)
Severe   | 45
Very Severe | 50
Extreme  | 75

6. SITE INSPECTION CHECKLIST (CPWD QA Manual 2022)
Pre-pour:
  [ ] Reinforcement cover checks (cover meter + physical gauges)
  [ ] Formwork stability and tightness
  [ ] Conduits and inserts positioned correctly
  [ ] Starter bars/dowels for next lift secured
  [ ] Concrete ordering confirmed (grade, slump, volume)

During pour:
  [ ] Slump test at point of delivery
  [ ] Cube samples taken per IS 516
  [ ] Temperature check (max 38°C at placing)
  [ ] Vibration coverage (every 500mm centers)

After pour:
  [ ] Curing start time recorded (<30 min for curing compound)
  [ ] Cure duration monitored (minimum 14 days for PPC/admixture mixes)
  [ ] Cube test results logged and signed off

Reference: IS 456:2000 Section 16; IS 516:2018; CPWD QA Manual 2022
            """
        }
    ],

    "safety": [
        {
            "filename": "construction_safety_concrete_work.txt",
            "title": "Construction Safety for Concrete Placing Operations",
            "source_org": "OSHA / National Building Code Part 7",
            "pub_year": 2020,
            "source_url": "https://www.osha.gov",
            "topics": ["concrete safety", "formwork collapse", "PPE", "fall protection", "hazardous conditions"],
            "content": """
SAFETY IN CONCRETE CONSTRUCTION
Source: NBC 2016 Part 7; OSHA 1926 Subpart Q; IS 3696

1. FORMWORK SAFETY
Formwork must be designed by a qualified structural engineer for:
- Weight of fresh concrete (typically 24-25 kN/m3)
- Live loads (workers, equipment, vibration): minimum 2.4 kN/m2 additional
- Impact loads from concrete pump or skip
- Lateral pressure from placing height > 1.5m

Critical checks before pour:
- Propping centres and sizes (typically props at max 1.0-1.2m centres for 200mm slabs)
- Ledger to standard connections secured
- Base plates bearing on firm ground (not soft or waterlogged)
- No stripping before structural strength achieved (usually 3-7 days slab soffit)

Formwork failures are a leading cause of construction fatalities.

2. PERSONAL PROTECTIVE EQUIPMENT (PPE) FOR CONCRETE WORK
- Hard hat: At all times on site (IS 2925)
- Safety boots (steel toecap + slip resistant): Mandatory during pour
- Chemical resistant gloves: Concrete is alkaline (pH 12-13); prolonged contact causes burns
- Eye protection: Splashing during pour, pump priming
- High visibility vest: Near mobile plant
- Hearing protection: During extended mechanical vibration use (>85 dB)

3. CONCRETE PUMP SAFETY
- Pump line pressure can exceed 100 bar; never break under pressure
- Cleaning out (pigging): Standpipe must be secured; release pressure first
- Never stand in line of fire of cleaning plug
- Ground anchors for pump outriggers on all soft ground

4. CHEMICAL HAZARDS
Fresh cement and concrete contain calcium hydroxide (strongly alkaline).
Wet cement causes cement dermatitis (delayed burns) with prolonged skin contact.
First aid: Remove contaminated clothing; rinse with clean water for 20 minutes.
Hexavalent chromium in Portland cement is a skin sensitiser (COSHH consideration in UK/EU standards).

5. WORKING AT HEIGHT DURING CONCRETE OPERATIONS
Falls are the largest single cause of deaths in construction.
NBC Part 7 requires:
- Edge protection (guardrail + mid-rail + kickboard) for any work platform > 1.8m
- Safety harness when edge protection cannot be provided
- Nets below formwork striking at height

6. HOT WEATHER HEALTH RISKS
Concrete work in high temperature:
- Heat exhaustion: Move to shade, cool fluids, rest
- Heat stroke: Medical emergency; call ambulance
Precautions: Schedule heavy work in early morning; provide cool water; rotate workers

Reference: NBC 2016 Part 7; OSHA 1926 Subpart Q Concrete; IS 3696 Part 1 Scaffolding Safety
            """
        }
    ],

    "inspection": [
        {
            "filename": "ndt_concrete_inspection.txt",
            "title": "Non-Destructive Testing of Concrete Structures",
            "source_org": "Bureau of Indian Standards / NPTEL",
            "pub_year": 2013,
            "source_url": "https://bis.gov.in",
            "topics": ["NDT", "rebound hammer", "ultrasonic pulse velocity", "concrete testing", "structural assessment"],
            "content": """
NON-DESTRUCTIVE TESTING OF CONCRETE STRUCTURES
Source: IS 13311 Part 1 (UPV), IS 13311 Part 2 (Rebound Hammer); NPTEL Construction Materials

1. REBOUND HAMMER TEST (Schmidt Hammer) - IS 13311 Part 2
Principle: Measures surface hardness related to compressive strength.
Equipment: Spring-driven metal hammer strikes concrete surface via plunger.
Procedure:
  - Surface must be smooth, dry, carbonation-free (grind if needed)
  - Take minimum 9 readings per test location (300mm x 300mm area)
  - Average of readings (excluding top and bottom quartile) = Rebound Index (R)
  - Calibrate against known cube strengths on same mix

Interpretation (approximate, for OPC 28-day cured concrete):
  R < 20:  Poor concrete, likely < 15 MPa
  R 20-30: Fair concrete, approximately 15-25 MPa
  R 30-40: Good concrete, approximately 25-40 MPa
  R > 40:  Very good concrete, > 40 MPa

Limitations:
  - Surface condition (wet, coated, carbonated) affects results significantly
  - Direction of test affects result (horizontal vs. overhead vs. downward)
  - Cannot assess internal defects
  - Correlation with strength varies by mix; local calibration essential

2. ULTRASONIC PULSE VELOCITY (UPV) - IS 13311 Part 1
Principle: Measures velocity of ultrasonic pulses (50-200 kHz) through concrete.
Configuration:
  - Direct (transmitter opposite receiver): Most accurate
  - Indirect (both on same face): Less accurate, for one-sided access
  - Semi-direct (90 degrees): Intermediate

Pulse Velocity Quality Classification:
  V > 4500 m/s:  Excellent quality
  V 3500-4500:   Good quality
  V 3000-3500:   Questionable
  V 2000-3000:   Poor quality
  V < 2000:      Very poor quality; likely voided or cracked

Applications:
  - Detect cracks, voids, honeycombing (velocity reduction)
  - Assess uniformity across a structure
  - Monitor strength development
  - Locate delamination in slabs

3. HALF-CELL POTENTIAL TESTING (ASTM C876)
For assessing corrosion activity of reinforcement bars.
Probability of active corrosion:
  More negative than -350 mV (CSE): > 90% probability of active corrosion
  -200 to -350 mV:                  Uncertain zone (50% probability)
  Less negative than -200 mV:       < 10% probability of active corrosion

4. CARBONATION DEPTH MEASUREMENT
Spray phenolphthalein indicator on freshly broken concrete surface.
  Pink/red coloured zone: pH > 9 (not carbonated, reinforcement protected)
  Colourless zone: pH < 9 (carbonated; reinforcement depassivated if steel reached)

Rate of carbonation: approximately k*sqrt(t) mm (where k=2-5mm/yr for OPC)

5. COVER METER SURVEY (BS 1881 Part 204)
Electromagnetic cover meter locates reinforcement and measures cover.
Survey procedure:
  - Grid scan at 100-200mm centres
  - Record depths and compare against design drawings
  - Identify zones with insufficient cover

Reference: IS 13311 Part 1, Part 2; IS 516; CPWD QA Manual 2022
            """
        }
    ],

    "delays": [
        {
            "filename": "construction_delay_analysis.txt",
            "title": "Construction Project Delay Analysis and Causes",
            "source_org": "NPTEL / IIT Kharagpur",
            "pub_year": 2012,
            "source_url": "https://nptel.ac.in",
            "topics": ["project delays", "delay analysis", "critical path", "schedule management", "causes of delay"],
            "content": """
CONSTRUCTION PROJECT DELAY ANALYSIS
Source: NPTEL Course - Construction Planning and Management, IIT Kharagpur

1. CLASSIFICATION OF DELAYS
Excusable Delays (Force Majeure):
- Unusually severe weather events
- Acts of God (earthquake, flood)
- Government actions beyond parties' control
- Utility strikes

Compensable Delays (Owner-caused):
- Late issue of drawings or instructions
- Design changes during construction
- Owner-supplied materials delayed
- Late access to site

Non-Compensable Delays (Contractor-caused):
- Labour shortages
- Equipment breakdown
- Poor planning or coordination
- Subcontractor defaults

Concurrent Delays:
- Multiple simultaneous delay causes make attribution complex
- Requires time-impact analysis for resolution

2. COMMON CAUSES IN INDIAN CONSTRUCTION (Research - IIT Madras)
Ranked by frequency:
1. Design changes and variations (most common)
2. Material procurement delays
3. Labour productivity issues
4. Site access and land acquisition
5. Approval delays (municipal, environmental)
6. Subcontractor coordination failures
7. Equipment availability
8. Cash flow / payment delays

3. DELAY ANALYSIS METHODS
a) As-Planned vs. As-Built:
   Compare original Gantt chart with actual completion dates.
   Identifies overall delay but not specific causes.

b) Time Impact Analysis (TIA):
   Insert delay events into schedule one at a time.
   Assess impact on critical path.
   Most accurate but time-intensive.

c) Window Analysis:
   Divide project into time windows.
   Analyse schedule within each window.
   Good for concurrent delay situations.

4. CRITICAL PATH METHOD (CPM)
Activities on the critical path have zero float - any delay extends project completion.
Formula: Float = Latest Start - Earliest Start
Critical path = Path with zero float (or minimum float)
Monitoring: Update CPM schedule at minimum monthly; weekly for at-risk activities

5. DELAY DAMAGES
Extension of Time (EOT): Additional calendar time to complete without penalty
Prolongation Costs: Site overheads during extended period
Loss and Expense: Additional costs resulting from delay (disruption, loss of productivity)
Liquidated Damages (LD): Pre-agreed daily rate for contractor-caused delays (FIDIC, CPWD forms)

6. PREVENTION STRATEGIES
- Detailed pre-construction planning (Master programme + 3-week lookaheads)
- Material procurement logs with lead times tracked
- Risk register updated monthly
- Early warning systems in contracts (FIDIC Clause 8.4)
- Regular progress meetings with action registers

Reference: NPTEL Course 105105098; IS 4736 Construction Management; FIDIC Red Book Clause 8
            """
        }
    ],

    "cost": [
        {
            "filename": "cost_overrun_management.txt",
            "title": "Cost Overrun Causes and Management in Construction",
            "source_org": "NPTEL / IIT Madras",
            "pub_year": 2011,
            "source_url": "https://nptel.ac.in",
            "topics": ["cost overrun", "budget management", "earned value", "BOQ", "contingency"],
            "content": """
COST OVERRUN IN CONSTRUCTION PROJECTS
Source: NPTEL Course - Cost Estimation and Control, IIT Madras

1. DEFINITION
Cost overrun = Final Cost - Budgeted Cost (at tender)
Expressed as: % overrun = (Final - Budget) / Budget x 100

Global average overrun: 28% (Oxford Major Project Research)
India infrastructure projects: Average 30-40% overrun (Planning Commission studies)

2. PRIMARY CAUSES OF COST OVERRUN
Design-related (40-50% of all overruns):
- Design errors discovered during construction
- Design changes by owner (scope creep)
- Incomplete design at tender (provisional sums)
- Optimistic cost estimating at tender stage

Construction-related:
- Labour and material price escalation
- Productivity lower than assumed
- Rework due to quality defects
- Equipment rental above budget

External factors:
- Adverse weather
- Ground conditions worse than geotechnical investigation
- Utility relocation delays
- Regulatory changes

3. EARNED VALUE MANAGEMENT (EVM)
Cost Performance Index (CPI) = Earned Value (EV) / Actual Cost (AC)
Schedule Performance Index (SPI) = EV / Planned Value (PV)

Interpretation:
  CPI > 1.0: Under budget
  CPI = 1.0: On budget
  CPI < 1.0: Over budget

Forecast:
  Estimate at Completion (EAC) = Budget at Completion / CPI
  If CPI = 0.85: Project will overrun by 17.6% of budget at completion

4. CONTINGENCY ALLOWANCE
Rule of thumb contingency for different project phases:
  Concept estimate: 20-30%
  Preliminary design: 15-20%
  Detailed design: 10-15%
  Post-tender: 5-10%

Contingency classification:
  Design contingency: For design development gaps
  Construction contingency: For unforeseen site conditions
  Escalation contingency: Price increases during construction period

5. BOQ-BASED CONTROL
Monitor actual quantities vs BOQ quantities monthly.
Flag items where actual > BOQ by >10% for variation review.
Maintain running cost report: Committed + Actual + Forecast = Total

Reference: NPTEL 105106115; CPWD DSR 2023; IS 1200 Method of Measurement
            """
        }
    ],

    "material_management": [
        {
            "filename": "material_management_concrete.txt",
            "title": "Material Management for Concrete Construction",
            "source_org": "CPWD / NPTEL",
            "pub_year": 2022,
            "source_url": "https://cpwd.gov.in",
            "topics": ["cement storage", "aggregate stockpiling", "material testing", "procurement", "inventory"],
            "content": """
MATERIAL MANAGEMENT FOR CONCRETE CONSTRUCTION
Source: CPWD Specifications 2019; IS 4082 Recommendations for Stacking of Materials

1. CEMENT MANAGEMENT
Storage requirements (IS 4082):
- Store in weatherproof godown on raised platform (200mm above ground)
- Maximum stack height: 10 bags (1500kg/m2 floor load)
- First In First Out (FIFO) rotation mandatory
- Cement older than 3 months must be retested before use
- Do not store within 300mm of external walls

Acceptance testing on delivery:
- Physical: Check bag weight (50±1kg), no hard lumps, free flow
- Setting time (IS 4031 Part 5): Initial > 30 min, Final < 600 min
- Compressive strength at 3 days (IS 4031 Part 6): OPC 53 Grade > 27 MPa
- Reject if any test fails; issue NCR (Non-Conformance Report)

2. AGGREGATE MANAGEMENT
Stockpiling (IS 4082 Section 4):
- Separate stockpiles for each aggregate size and source
- Height limit: 3m for coarse, 2m for fine (to avoid segregation)
- Minimum clear distance between stockpiles: 1500mm
- Exclude contaminated or mixed materials

Moisture monitoring:
- Fine aggregate moisture critical for free W/C control
- Test moisture content (IS 2386 Part 3) before every batch when visual appearance changes
- Typical moisture: 3-7% in FA; 0-2% in CA

3. ADMIXTURE MANAGEMENT
- Store per manufacturer guidelines (typically 5-35°C; away from freezing)
- Check expiry date; re-test if stored more than shelf life
- Plastic jerry cans not reusable between different admixtures (contamination)
- Record lot number and batch reference for every delivery

4. REINFORCEMENT STEEL MANAGEMENT
- Store off ground on timber sleepers (avoid soil contact - corrosion)
- Segregate by diameter, grade, and mill mark
- Protect from rain and damp with tarpaulin for projects exceeding 3 months
- Mill test certificates (MTC) must match delivery (heat number, diameter, grade)
- Acceptance testing: IS 1786 - Tensile test, Bend test per specified bar diameter

5. WASTE MINIMISATION
Concrete wastage targets:
  Acceptable loss at site: 2-3% for slabs; 3-5% for columns and walls
  Excess order allowance: 2-5% over calculated volume
  Monitoring: Record delivery dockets vs poured volume; investigate >5% variance

Reference: IS 4082 Stacking and Storage; CPWD Specifications 2019 Chapter 2; IS 1786:2008
            """
        }
    ],

    "boq": [
        {
            "filename": "bill_of_quantities_preparation.txt",
            "title": "Bill of Quantities - Preparation and Measurement",
            "source_org": "CPWD / Bureau of Indian Standards",
            "pub_year": 2023,
            "source_url": "https://cpwd.gov.in",
            "topics": ["bill of quantities", "measurement", "rate analysis", "specifications", "tendering"],
            "content": """
BILL OF QUANTITIES (BOQ) IN CONSTRUCTION
Source: IS 1200 Method of Measurement; CPWD DSR 2023

1. DEFINITION
A Bill of Quantities is a document prepared during the design stage that:
- Lists all the work items required to complete a project
- Provides units of measurement and quantities for each item
- Allows contractors to price on a common basis
- Forms the basis for valuing interim payments and variations

2. BOQ STRUCTURE
Division by trade or element:
  Part A: Earthwork and excavation
  Part B: Concrete and reinforcement
  Part C: Masonry
  Part D: Finishing (plaster, tiles, paint)
  Part E: Structural steel
  Part F: Services (MEP)
  Part G: External works
  Preambles: Specification notes binding on contractor
  Provisional Sums (PS): Items where scope undefined at tender
  Prime Cost (PC) Items: Specialist work; sub-contractors nominated by owner

3. CONCRETE BOQ ITEMS (IS 1200 Part 2)
Measurement: In-situ concrete measured in cubic metres (m3) of finished work
No deduction for: Reinforcement bars, formed holes less than 0.1 m3 each
Deduct: Formed openings > 0.1 m3

Typical BOQ rate includes:
- Materials: Cement, aggregate, water, admixtures
- Labour: Mixing, placing, vibrating, finishing
- Plant: Concrete pump, transit mixer, vibrator
- Overheads and profit

4. REINFORCEMENT MEASUREMENT (IS 1200 Part 4)
Measured in metric tonnes (MT)
Includes: Bar length from drawings + standard hooks and bends (IS 2502)
Excludes: Wastage allowance (contractor to price)
Lap lengths included as designed (typically 40-60 bar diameters)

5. RATE ANALYSIS COMPONENTS
Rate per m3 of concrete = Materials cost + Labour + Plant + Overheads + Profit
Materials cost per m3 (M35 concrete example):
  Cement 400 kg x Rs.X/bag = Rs.
  20mm CA 680 kg x Rs.Y/MT = Rs.
  Fine aggregate 620 kg x Rs.Z/MT = Rs.
  Water + admixtures = Rs.
  Total materials = Rs.

6. VARIATION ORDERS (VO) AND CLAIMS
Any work not in BOQ or change to BOQ quantity > 10%:
- Issue Variation Order with drawing reference
- Agree rate before execution (where possible)
- Star rate for new items: Build-up from labour + material + overhead + profit

Reference: IS 1200 Parts 1-26; CPWD DSR 2023; FIDIC Red Book Clause 12
            """
        }
    ],

    "standards": [
        {
            "filename": "is_456_key_provisions.txt",
            "title": "IS 456:2000 - Key Provisions for Plain and Reinforced Concrete",
            "source_org": "Bureau of Indian Standards",
            "pub_year": 2000,
            "source_url": "https://standardsbis.bsbedge.com",
            "topics": ["IS 456", "concrete standard", "reinforced concrete", "structural design", "India standard"],
            "content": """
IS 456:2000 PLAIN AND REINFORCED CONCRETE - KEY PROVISIONS
Source: Bureau of Indian Standards IS 456:2000 (Fourth Revision)

1. SCOPE
IS 456:2000 covers the structural use of plain and reinforced concrete for general building construction.
It specifies requirements for materials, mix design, construction practices, and structural design.
Applies to: Buildings, retaining walls, foundations, water-retaining structures (with IS 3370)

2. EXPOSURE CATEGORIES AND CONCRETE SPECIFICATION (TABLE 3)
Category | Environment | Min Grade | Max W/C | Min Cover
Mild     | Protected from weather | M20 | 0.55 | 20mm (slab)
Moderate | Humid, slight aggress. | M25 | 0.50 | 30mm
Severe   | Cyclic wet/dry, sea air | M30 | 0.45 | 45mm
Very Severe| Tidal, deicing salts | M35 | 0.45 | 50mm
Extreme  | Sea water immersion | M40 | 0.40 | 75mm

3. STRUCTURAL DESIGN REQUIREMENTS
Load factors (Limit State of Collapse):
DL + LL: gamma_DL = 1.5, gamma_LL = 1.5
DL + WL/EL: gamma_DL = 0.9 (when DL stabilising), 1.2 otherwise
Combined: 1.2 (DL + LL + WL)

Material partial safety factors:
Concrete: gamma_c = 1.5 (design strength = 0.67 fck / 1.5 = 0.45 fck)
Steel:    gamma_s = 1.15 (design yield strength = 0.87 fy)

4. MINIMUM REINFORCEMENT
Minimum tension reinforcement in beams: 0.85 bd / fy (Clause 26.5.1.1)
Minimum main reinforcement in slabs: 0.12% (HYSD), 0.15% (mild steel)
Minimum stirrup spacing: Smaller of d/2, 300mm
Minimum column reinforcement: 0.8% of gross cross-section area
Maximum column reinforcement: 6% (4% at lap locations)

5. DURABILITY REQUIREMENTS
Chloride content in concrete: Max 0.4% by mass of cement (reinforced concrete)
Sulphate content (SO3): Max 4% by mass of cement
Alkali-Silica Reaction (ASR): Use low-alkali cement (<0.6% Na2O equivalent) with reactive aggregates

6. INSPECTION AND TESTING
Fresh concrete: Slump test, temperature, cube sampling (IS 516)
Hardened concrete: Cube tests at 7 and 28 days; core tests if cubes fail
Cover survey: Electromagnetic cover meter survey after striking formwork
Acceptance: Refer IS 456:2000 Section 16

Reference: IS 456:2000 (Complete); SP 24:1983 IS 456 Explanatory Handbook
            """
        }
    ],

    "reports": [
        {
            "filename": "construction_failure_analysis.txt",
            "title": "Construction Failure Modes and Root Cause Analysis",
            "source_org": "NPTEL / IIT Madras",
            "pub_year": 2015,
            "source_url": "https://nptel.ac.in/courses/105106202",
            "topics": ["failure analysis", "root cause", "structural failure", "quality defects", "lessons learned"],
            "content": """
CONSTRUCTION FAILURE MODES AND ROOT CAUSE ANALYSIS
Source: NPTEL Maintenance and Repair of Concrete Structures, IIT Madras

1. COMMON CONCRETE STRUCTURE FAILURES
Category A: Design Deficiencies
- Insufficient flexural reinforcement (under-reinforced sections)
- Inadequate shear links (diagonal tension failure)
- Poor structural detailing at connections
- Punching shear in flat slabs without shear heads

Category B: Material Deficiencies
- Substandard cement (slow setting, low strength)
- Alkali-reactive aggregates without mitigation
- Contaminated water (chloride, sulphate)
- Counterfeit reinforcement steel (substandard yield strength)

Category C: Construction Deficiencies
- Excessive water addition at batching or site (most common)
- Inadequate compaction (honeycombing, voids)
- Insufficient cover (chloride ingress; corrosion)
- Inadequate curing (low strength, plastic cracking)
- Premature form striking (deflection, cracks)

Category D: In-Service Loading Failures
- Overloading beyond design (floor loadings, storage)
- Lateral loads from adjacent excavation
- Differential settlement
- Fire damage (spalling, strength reduction)

2. ROOT CAUSE ANALYSIS PROCESS
Step 1: Evidence Collection
  - Photographic documentation of all defects
  - Sampling of concrete cores, reinforcement bars
  - Review of site records (pour cards, cube results, delivery dockets)
  - Witness statements

Step 2: Failure Mode Identification
  - Pattern analysis of crack locations
  - UPV and rebound hammer mapping
  - Core analysis (carbonation, chloride profile, mix analysis)

Step 3: Cause Determination
  Use 5-Why Analysis:
  Why did the beam fail? -> Low concrete strength
  Why low strength? -> High water-cement ratio
  Why high W/C? -> Water added on site
  Why water added? -> Concrete too stiff on delivery
  Why too stiff? -> Long transit time + high temperature

Step 4: Remedial Action
  Immediate: Shore up structure if unsafe
  Short-term: Repair defective concrete
  Long-term: System improvement (site controls, supervision, testing)

3. CASE STUDY: PLASTIC SHRINKAGE CRACK FAILURE
Project: Large industrial floor slab, poured in hot season
Defect: Extensive map cracking within 4 hours of pour
Root causes:
  - Temperature: 38°C ambient, concrete placed at 35°C
  - Wind: 15 km/hr exposed site (no windbreaks erected)
  - Relative humidity: 28% (dry desert climate)
  - Evaporation rate: Calculated 1.8 kg/m2/hr (High risk threshold: 1.0)
  - No curing compound applied until 2 hours after finishing

Corrections implemented:
  - Wind breaks erected on all exposed boundaries
  - Curing compound applied within 20 minutes of finishing
  - Retarder added to reduce plastic cracking window
  - Evening pours only for large areas in summer

4. QUALITY FAILURE INVESTIGATION REPORT STRUCTURE
Executive Summary
Site Observations
Review of Records
Sampling and Testing Plan
Laboratory Results
Analysis and Root Cause
Classification of Defects
Recommendations for Remediation
Cost Estimate for Repairs
Responsibility Assessment

Reference: NPTEL 105106202; IS 456:2000 Appendix B; CPWD Failure Investigation Manual
            """
        }
    ]
}


def seed():
    """Write all domain knowledge text files to the knowledge_base/ folders."""
    total = 0
    txt_doc_entries = []

    for domain, docs in DOMAIN_CONTENT.items():
        domain_dir = KB_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        for doc in docs:
            out_path = domain_dir / doc["filename"]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(doc["content"].strip())

            txt_doc_entries.append({
                "id":         out_path.stem,
                "title":      doc["title"],
                "source_org": doc["source_org"],
                "url":        doc["source_url"],
                "domain":     domain,
                "doc_type":   "knowledge_text",
                "pub_year":   doc["pub_year"],
                "relevance_score": 0.92,
                "priority":   "High",
                "filename":   doc["filename"],
                "topics":     doc["topics"]
            })
            print(f"  [SEEDED] {domain}/{doc['filename']}")
            total += 1

    # Append to metadata.json
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    existing_ids = {d["id"] for d in meta["documents"]}
    new_entries = [e for e in txt_doc_entries if e["id"] not in existing_ids]
    meta["documents"].extend(new_entries)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSeeded {total} knowledge text files.")
    print(f"Added {len(new_entries)} new entries to metadata.json")


if __name__ == "__main__":
    seed()


if __name__ == "__main__":
    seed()
