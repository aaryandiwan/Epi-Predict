"""
Epi Predict – Public Awareness & Health Education Module

Provides comprehensive, medically-reviewed influenza information for
the dashboard's public-facing education pages.  All content is curated
from WHO, CDC, and India NCDC guidelines.

Sections:
    • Flu information (symptoms, types, transmission)
    • Preventive measures
    • Vaccination guidance (schedule, eligibility, side effects)
    • Emergency contacts (WHO, CDC, India NCDC, helplines)
    • Dynamic seasonal advisory (risk-level-aware)

Author : Epi Predict Team
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from config.settings import RISK_THRESHOLDS

logger = logging.getLogger("epi_predict.public_awareness")


# ─── Flu Information ─────────────────────────────────────────────────────────

def get_flu_info() -> Dict[str, Any]:
    """Return comprehensive influenza information.

    Returns:
        Dictionary with keys:
            - **overview** (str): Brief introduction.
            - **symptoms** (dict): Common, severe, and emergency symptoms.
            - **types** (list[dict]): Influenza virus types (A, B, C, D).
            - **transmission** (dict): How influenza spreads.
            - **incubation_period** (str): Typical incubation window.
            - **contagious_period** (str): When patients are contagious.
            - **high_risk_groups** (list[str]): Populations at elevated risk.

    Example::

        >>> info = get_flu_info()
        >>> info["symptoms"]["common"]
        ['Fever (100.4°F / 38°C or higher)', ...]
    """
    info: Dict[str, Any] = {
        "overview": (
            "Influenza (flu) is a contagious respiratory illness caused by "
            "influenza viruses that infect the nose, throat, and sometimes "
            "the lungs. It can cause mild to severe illness, and in serious "
            "cases can lead to hospitalisation or death. The best way to "
            "prevent flu is by getting vaccinated each year."
        ),
        "symptoms": {
            "common": [
                "Fever (100.4°F / 38°C or higher)",
                "Chills and sweats",
                "Headache and muscle aches",
                "Persistent dry cough",
                "Sore throat and runny or stuffy nose",
                "Fatigue and general weakness",
                "Loss of appetite",
            ],
            "severe": [
                "Shortness of breath or difficulty breathing",
                "Persistent chest pain or pressure",
                "Severe vomiting or inability to keep liquids down",
                "Sudden dizziness or confusion",
                "Bluish discoloration of lips or face (cyanosis)",
            ],
            "emergency_warning_signs": [
                "Difficulty breathing or shortness of breath at rest",
                "Persistent high fever (>103°F / 39.4°C) for 3+ days",
                "Severe chest pain",
                "Sudden confusion or altered mental state",
                "Seizures",
                "Inability to urinate (dehydration indicator)",
                "Symptoms that improve then return with worsening cough and fever",
            ],
        },
        "types": [
            {
                "type": "Influenza A",
                "description": (
                    "The most common and virulent type. Responsible for "
                    "seasonal epidemics and occasional pandemics. Found in "
                    "humans, birds, pigs, horses, and other animals."
                ),
                "subtypes": "H1N1, H3N2, H5N1 (avian), H7N9",
                "severity": "Moderate to Severe",
            },
            {
                "type": "Influenza B",
                "description": (
                    "Causes seasonal epidemics but is generally less severe "
                    "than Type A. Found only in humans. Two lineages: "
                    "B/Yamagata and B/Victoria."
                ),
                "subtypes": "B/Yamagata, B/Victoria",
                "severity": "Mild to Moderate",
            },
            {
                "type": "Influenza C",
                "description": (
                    "Causes mild respiratory illness and is not thought to "
                    "cause epidemics. Detected in humans and pigs."
                ),
                "subtypes": "N/A",
                "severity": "Mild",
            },
            {
                "type": "Influenza D",
                "description": (
                    "Primarily affects cattle and is not known to infect or "
                    "cause illness in humans."
                ),
                "subtypes": "N/A",
                "severity": "Not applicable to humans",
            },
        ],
        "transmission": {
            "primary_routes": [
                "Respiratory droplets from coughing, sneezing, or talking (within ~6 feet)",
                "Airborne aerosols in enclosed, poorly ventilated spaces",
                "Direct contact with an infected person (e.g. handshake, hug)",
                "Touching contaminated surfaces (fomites) then touching eyes, nose, or mouth",
            ],
            "survival_on_surfaces": (
                "Influenza virus can survive on hard surfaces (doorknobs, "
                "phones, handrails) for 24-48 hours and on soft surfaces "
                "(tissues, clothing) for 8-12 hours."
            ),
            "environmental_factors": (
                "Transmission is enhanced in cold, dry conditions. "
                "Humidity below 40% allows respiratory droplets to remain "
                "airborne longer, contributing to seasonal winter peaks."
            ),
        },
        "incubation_period": (
            "1 to 4 days (average 2 days) after exposure to the virus."
        ),
        "contagious_period": (
            "Infected individuals can spread the virus from 1 day before "
            "symptoms appear to 5-7 days after becoming sick. Young children "
            "and immunocompromised individuals may be contagious for longer."
        ),
        "high_risk_groups": [
            "Adults aged 65 years and older",
            "Children younger than 5 years (especially under 2 years)",
            "Pregnant women and women up to 2 weeks postpartum",
            "Residents of nursing homes and long-term care facilities",
            "People with chronic medical conditions (asthma, diabetes, heart disease)",
            "Immunocompromised individuals (HIV/AIDS, cancer treatment, organ transplant)",
            "Healthcare workers and frontline responders",
            "People with a BMI ≥ 40 (severe obesity)",
        ],
    }

    logger.info("Flu information retrieved successfully.")
    return info


# ─── Preventive Measures ────────────────────────────────────────────────────

def get_preventive_measures() -> List[Dict[str, Any]]:
    """Return a prioritised list of influenza prevention strategies.

    Returns:
        List of dicts, each with:
            - **measure** (str): Title of the preventive action.
            - **description** (str): Detailed guidance.
            - **effectiveness** (str): Qualitative effectiveness rating.
            - **category** (str): Grouping (vaccination, hygiene, lifestyle,
              environmental).

    Example::

        >>> measures = get_preventive_measures()
        >>> measures[0]["measure"]
        'Annual Vaccination'
    """
    measures: List[Dict[str, Any]] = [
        {
            "measure": "Annual Vaccination",
            "description": (
                "Get the seasonal influenza vaccine every year. The vaccine "
                "is updated annually to match circulating strains and is the "
                "single most effective prevention method."
            ),
            "effectiveness": "High (40-60% reduction in flu illness)",
            "category": "vaccination",
        },
        {
            "measure": "Hand Hygiene",
            "description": (
                "Wash hands frequently with soap and water for at least "
                "20 seconds, especially after coughing, sneezing, or "
                "touching public surfaces. Use alcohol-based hand sanitizer "
                "(≥60% alcohol) when soap is not available."
            ),
            "effectiveness": "High",
            "category": "hygiene",
        },
        {
            "measure": "Respiratory Etiquette",
            "description": (
                "Cover mouth and nose with a tissue or elbow when coughing "
                "or sneezing. Dispose of tissues immediately and wash hands "
                "afterwards."
            ),
            "effectiveness": "Moderate to High",
            "category": "hygiene",
        },
        {
            "measure": "Mask Wearing",
            "description": (
                "Wear a well-fitted surgical or N95 mask in crowded or "
                "enclosed spaces during peak flu season, especially if you "
                "are in a high-risk group or caring for sick individuals."
            ),
            "effectiveness": "Moderate to High",
            "category": "hygiene",
        },
        {
            "measure": "Avoid Close Contact",
            "description": (
                "Maintain at least 1 metre (3 feet) distance from people "
                "who are visibly ill. Avoid crowded indoor spaces during "
                "outbreaks."
            ),
            "effectiveness": "Moderate",
            "category": "environmental",
        },
        {
            "measure": "Surface Disinfection",
            "description": (
                "Regularly clean and disinfect frequently touched surfaces – "
                "doorknobs, light switches, phones, keyboards, and countertops "
                "– using EPA-registered disinfectants or diluted bleach solution."
            ),
            "effectiveness": "Moderate",
            "category": "environmental",
        },
        {
            "measure": "Adequate Ventilation",
            "description": (
                "Ensure indoor spaces are well ventilated. Open windows when "
                "possible or use HEPA air purifiers to reduce airborne viral "
                "concentration."
            ),
            "effectiveness": "Moderate",
            "category": "environmental",
        },
        {
            "measure": "Healthy Lifestyle",
            "description": (
                "Maintain a balanced diet rich in fruits and vegetables, "
                "exercise regularly (≥150 min/week), get 7-9 hours of sleep, "
                "manage stress, and stay hydrated to support immune function."
            ),
            "effectiveness": "Supportive",
            "category": "lifestyle",
        },
        {
            "measure": "Stay Home When Sick",
            "description": (
                "If you develop flu symptoms, stay home for at least 24 hours "
                "after your fever resolves (without fever-reducing medication) "
                "to prevent spreading the virus to others."
            ),
            "effectiveness": "High",
            "category": "hygiene",
        },
        {
            "measure": "Antiviral Prophylaxis",
            "description": (
                "In high-risk situations, consult a healthcare provider about "
                "antiviral medications (oseltamivir / Tamiflu) for pre- or "
                "post-exposure prophylaxis, especially for contacts of "
                "confirmed cases."
            ),
            "effectiveness": "High (when taken within 48 hours)",
            "category": "medical",
        },
    ]

    logger.info("Preventive measures retrieved: %d items.", len(measures))
    return measures


# ─── Vaccination Guidance ────────────────────────────────────────────────────

def get_vaccination_guidance() -> Dict[str, Any]:
    """Return detailed influenza vaccination guidance.

    Returns:
        Dictionary covering schedule, eligibility, types, side effects,
        and where to get vaccinated.
    """
    guidance: Dict[str, Any] = {
        "overview": (
            "Seasonal influenza vaccination is the most effective method to "
            "prevent flu and its complications. The WHO recommends annual "
            "vaccination, ideally before the onset of flu season."
        ),
        "schedule": {
            "northern_hemisphere": (
                "September – November (before flu season peaks in "
                "December – February). Vaccination can still be beneficial "
                "if received later in the season."
            ),
            "southern_hemisphere": (
                "March – May (before flu season peaks in June – August)."
            ),
            "india_specific": (
                "India experiences two flu seasons: post-monsoon (August – "
                "October) in parts of western and southern India, and "
                "winter (November – February) in northern India. Vaccination "
                "is recommended before each respective season."
            ),
            "children": (
                "Children aged 6 months – 8 years receiving the flu vaccine "
                "for the first time need two doses, given at least 4 weeks "
                "apart. One dose annually thereafter."
            ),
        },
        "eligibility": {
            "recommended_for": [
                "Everyone aged 6 months and older",
                "Pregnant women (any trimester)",
                "Healthcare workers and hospital staff",
                "Elderly (≥65 years)",
                "People with chronic diseases (diabetes, asthma, heart disease, kidney disease)",
                "Immunocompromised individuals",
                "Children aged 6 months to 5 years",
                "Caregivers of high-risk individuals",
            ],
            "contraindications": [
                "Children younger than 6 months",
                "Individuals with a severe, life-threatening allergy to the flu vaccine or any ingredient",
                "People who have had Guillain-Barré Syndrome (GBS) within 6 weeks of a prior flu vaccine",
            ],
        },
        "vaccine_types": [
            {
                "name": "Inactivated Influenza Vaccine (IIV)",
                "route": "Intramuscular injection",
                "approved_for": "6 months and older",
                "notes": "Most commonly used; available as trivalent or quadrivalent.",
            },
            {
                "name": "Live Attenuated Influenza Vaccine (LAIV)",
                "route": "Nasal spray",
                "approved_for": "2 – 49 years (non-pregnant)",
                "notes": "Not recommended for immunocompromised individuals.",
            },
            {
                "name": "Recombinant Influenza Vaccine (RIV)",
                "route": "Intramuscular injection",
                "approved_for": "18 years and older",
                "notes": "Egg-free option suitable for people with egg allergies.",
            },
            {
                "name": "High-Dose Influenza Vaccine",
                "route": "Intramuscular injection",
                "approved_for": "65 years and older",
                "notes": "Contains 4× the antigen of standard dose for stronger immune response.",
            },
        ],
        "side_effects": {
            "common": [
                "Pain, redness, or swelling at injection site",
                "Low-grade fever (usually resolves in 1-2 days)",
                "Mild headache and muscle aches",
                "Fatigue and general malaise",
                "Nausea (rare)",
            ],
            "rare": [
                "Allergic reaction (anaphylaxis) – extremely rare (~1 in 1 million)",
                "Guillain-Barré Syndrome (GBS) – very rare (~1-2 per million vaccinated)",
                "Febrile seizures in young children (rare and generally benign)",
            ],
            "note": (
                "The flu vaccine CANNOT cause influenza. Inactivated vaccines "
                "contain no live virus, and nasal spray vaccines contain "
                "weakened virus that cannot cause flu illness."
            ),
        },
        "locations": {
            "general": [
                "Primary healthcare centres and government hospitals",
                "Private hospitals and clinics",
                "Pharmacies and drugstores (in many countries)",
                "Community health drives and vaccination camps",
                "Workplace vaccination programmes",
            ],
            "india_specific": [
                "Government Primary Health Centres (PHCs)",
                "District and taluk hospitals",
                "Accredited private hospitals (Apollo, Fortis, Max, etc.)",
                "National Immunization Programme centres",
                "CGHS and ECHS dispensaries for government employees",
            ],
            "finder_urls": {
                "WHO": "https://www.who.int/teams/immunization-vaccines-and-biologicals",
                "CDC_VaccineFinder": "https://www.vaccines.gov/",
                "India_CoWIN": "https://www.cowin.gov.in/",
            },
        },
    }

    logger.info("Vaccination guidance retrieved successfully.")
    return guidance


# ─── Emergency Contacts ─────────────────────────────────────────────────────

def get_emergency_contacts() -> Dict[str, Any]:
    """Return emergency contact information for influenza response.

    Returns:
        Dictionary of organisations with phone numbers, websites,
        and descriptions.
    """
    contacts: Dict[str, Any] = {
        "international": [
            {
                "organisation": "World Health Organization (WHO)",
                "description": "Global health authority coordinating international influenza surveillance.",
                "website": "https://www.who.int/",
                "phone": "+41 22 791 21 11",
                "email": "mediainquiries@who.int",
                "services": [
                    "Global disease outbreak alerts",
                    "International Health Regulations",
                    "Technical guidance and situation reports",
                ],
            },
            {
                "organisation": "Centers for Disease Control and Prevention (CDC)",
                "description": "US national public health institute; leading global flu surveillance authority.",
                "website": "https://www.cdc.gov/flu/",
                "phone": "1-800-232-4636 (CDC-INFO)",
                "email": "cdcinfo@cdc.gov",
                "services": [
                    "FluView surveillance reports",
                    "Vaccination guidance",
                    "Antiviral treatment guidelines",
                ],
            },
        ],
        "india": [
            {
                "organisation": "National Centre for Disease Control (NCDC)",
                "description": (
                    "India's premier institute for disease surveillance, "
                    "investigation, and response."
                ),
                "website": "https://ncdc.mohfw.gov.in/",
                "phone": "+91-11-23909104",
                "email": "nilokheri.ncdc@gmail.com",
                "services": [
                    "Integrated Disease Surveillance Programme (IDSP)",
                    "Outbreak investigation and response",
                    "Weekly disease outbreak reports",
                ],
            },
            {
                "organisation": "Indian Council of Medical Research (ICMR)",
                "description": (
                    "Apex body for biomedical research and influenza "
                    "laboratory network in India."
                ),
                "website": "https://www.icmr.gov.in/",
                "phone": "+91-11-26588980",
                "email": "icmrhqds@sansad.nic.in",
                "services": [
                    "National Influenza Centre",
                    "Laboratory confirmation of influenza",
                    "Research and epidemiological studies",
                ],
            },
        ],
        "helplines": [
            {
                "name": "India Health Helpline",
                "number": "104",
                "description": "24/7 health information and medical advice helpline.",
                "availability": "24 hours, 7 days a week",
            },
            {
                "name": "India Emergency Services",
                "number": "112",
                "description": "Unified emergency response number for police, fire, and ambulance.",
                "availability": "24 hours, 7 days a week",
            },
            {
                "name": "India Ambulance Service",
                "number": "108",
                "description": "Free emergency ambulance service available in most states.",
                "availability": "24 hours, 7 days a week",
            },
            {
                "name": "WHO Health Alert (WhatsApp)",
                "number": "+41 79 893 18 92",
                "description": "Send 'hi' to get latest WHO health information via WhatsApp.",
                "availability": "Automated, always available",
            },
        ],
    }

    logger.info("Emergency contacts retrieved successfully.")
    return contacts


# ─── Seasonal Advisory ───────────────────────────────────────────────────────

def get_seasonal_advisory(risk_level: str) -> str:
    """Generate a dynamic public advisory based on the current risk level.

    Produces a multi-paragraph advisory text suitable for display on
    the dashboard home page or inclusion in public communications.

    Args:
        risk_level: Current risk classification key
            (``"low"`` | ``"moderate"`` | ``"high"`` | ``"severe"``).

    Returns:
        Multi-paragraph advisory string.

    Raises:
        ValueError: If *risk_level* is not recognised.
    """
    level = risk_level.strip().lower()
    threshold = RISK_THRESHOLDS.get(level)

    if threshold is None:
        valid = ", ".join(RISK_THRESHOLDS.keys())
        logger.error("Unknown risk level '%s'. Valid: %s", risk_level, valid)
        raise ValueError(
            f"Unknown risk level '{risk_level}'. Expected one of: {valid}"
        )

    icon = threshold["icon"]
    label = threshold["label"]

    _advisories: Dict[str, str] = {
        "low": (
            f"{icon} SEASONAL INFLUENZA ADVISORY – {label.upper()}\n\n"
            "Current influenza activity is within normal seasonal baseline "
            "levels. No unusual patterns have been detected in recent "
            "surveillance data.\n\n"
            "Recommended Actions:\n"
            "• Continue routine hand hygiene practices.\n"
            "• Ensure your seasonal flu vaccination is up to date.\n"
            "• Maintain a healthy lifestyle with regular exercise, balanced "
            "nutrition, and adequate sleep.\n"
            "• Stay informed through official public health channels.\n\n"
            "This is a good time to check that household members, especially "
            "young children and elderly relatives, are vaccinated for the "
            "current season."
        ),
        "moderate": (
            f"{icon} SEASONAL INFLUENZA ADVISORY – {label.upper()}\n\n"
            "Influenza activity has risen above baseline levels. An "
            "increased number of cases has been reported in surveillance "
            "networks, indicating the onset of a seasonal wave.\n\n"
            "Recommended Actions:\n"
            "• Wear a mask in crowded indoor settings (public transport, "
            "markets, offices).\n"
            "• Increase hand-washing frequency and carry hand sanitizer.\n"
            "• Monitor yourself and family members for flu symptoms "
            "(fever, cough, body aches).\n"
            "• Avoid close contact with individuals showing respiratory "
            "symptoms.\n"
            "• If you have not yet been vaccinated this season, do so now.\n\n"
            "Individuals in high-risk groups (elderly, pregnant women, "
            "immunocompromised) should exercise heightened caution and "
            "consult their healthcare provider if symptoms develop."
        ),
        "high": (
            f"{icon} SEASONAL INFLUENZA ADVISORY – {label.upper()}\n\n"
            "Significant influenza outbreak activity has been detected. "
            "Hospital admissions related to influenza-like illness (ILI) are "
            "elevated, and community transmission is widespread.\n\n"
            "Recommended Actions:\n"
            "• AVOID large gatherings, crowded events, and non-essential "
            "public outings.\n"
            "• GET VACCINATED if you have not already done so.\n"
            "• Stock up on essential medications (paracetamol, ORS, "
            "prescribed inhalers).\n"
            "• Limit use of public transport; walk, cycle, or use private "
            "vehicles where possible.\n"
            "• Ensure adequate ventilation in workplaces and homes.\n"
            "• Seek medical attention promptly if flu symptoms develop, "
            "especially if you are in a high-risk group.\n\n"
            "Employers are encouraged to activate flexible work-from-home "
            "policies and ensure sick leave provisions are adequate."
        ),
        "severe": (
            f"{icon} SEASONAL INFLUENZA ADVISORY – {label.upper()}\n\n"
            "A SEVERE influenza outbreak is in progress. This is an "
            "emergency-level public health situation requiring immediate "
            "action from individuals, communities, and institutions.\n\n"
            "Recommended Actions:\n"
            "• WORK FROM HOME wherever possible to reduce community "
            "transmission.\n"
            "• AVOID ALL unnecessary travel – defer domestic and "
            "international trips.\n"
            "• SEEK IMMEDIATE medical consultation if you experience fever, "
            "cough, difficulty breathing, or any emergency warning signs.\n"
            "• FOLLOW all government directives – comply with quarantine "
            "zones, travel restrictions, and public gathering bans.\n"
            "• PREPARE an emergency supply kit: 7 days of food, water, "
            "medications, masks, sanitizer, and hygiene essentials.\n"
            "• PROTECT vulnerable household members – isolate sick "
            "individuals in a separate, ventilated room.\n\n"
            "Healthcare facilities may be operating at high capacity. "
            "Use telemedicine for initial consultations where available. "
            "Contact emergency services (112 / 108) only for "
            "life-threatening situations.\n\n"
            "Stay informed via official channels:\n"
            "• NCDC: https://ncdc.mohfw.gov.in/\n"
            "• WHO: https://www.who.int/\n"
            "• CDC: https://www.cdc.gov/flu/"
        ),
    }

    advisory = _advisories.get(level, "No advisory available for this risk level.")

    logger.info(
        "Seasonal advisory generated for risk level '%s' (%d chars).",
        level,
        len(advisory),
    )

    return advisory
