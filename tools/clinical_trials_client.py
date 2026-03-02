"""
ClinicalTrials.gov Client

API client for ClinicalTrials.gov v2 API.
"""

import re
from typing import Optional, Dict, Any, List
import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov API v2.

    Features:
    - NCT ID extraction from URLs
    - Study details fetching
    - Search by condition/intervention
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self, timeout: float = 30.0):
        """
        Initialize ClinicalTrials.gov client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        logger.info("ClinicalTrials.gov client initialized")

    def extract_nct_id(self, url: str) -> Optional[str]:
        """
        Extract NCT ID from URL.

        Args:
            url: ClinicalTrials.gov URL

        Returns:
            NCT ID (e.g., "NCT03785249") or None
        """
        # NCT ID pattern: NCT followed by 8 digits
        match = re.search(r'(NCT\d{8})', url, re.IGNORECASE)
        if match:
            nct_id = match.group(1).upper()
            logger.debug("Extracted NCT ID", url=url, nct_id=nct_id)
            return nct_id
        return None

    async def fetch_study(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch study details by NCT ID.

        Args:
            nct_id: NCT ID (e.g., "NCT03785249")

        Returns:
            Study data dictionary
        """
        url = f"{self.BASE_URL}/studies/{nct_id}"

        logger.debug("Fetching clinical trial", nct_id=nct_id)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.warning(
                        "ClinicalTrials.gov API failed",
                        nct_id=nct_id,
                        status=response.status_code
                    )
                    return None

                data = response.json()
                protocol = data.get('protocolSection', {})

                # Extract key modules
                result = self._parse_protocol(nct_id, protocol)
                logger.debug("Clinical trial fetched", nct_id=nct_id, title=result.get('title', '')[:50])
                return result

            except Exception as e:
                logger.error("ClinicalTrials.gov API error", nct_id=nct_id, error=str(e))
                return None

    def _parse_protocol(self, nct_id: str, protocol: Dict) -> Dict[str, Any]:
        """
        Parse protocol section into structured data.

        Args:
            nct_id: NCT ID
            protocol: Protocol section from API

        Returns:
            Parsed study data
        """
        # Identification module
        ident = protocol.get('identificationModule', {})
        title = ident.get('officialTitle', ident.get('briefTitle', ''))
        org_study_id = ident.get('orgStudyIdInfo', {}).get('id', '')

        # Description module
        desc = protocol.get('descriptionModule', {})
        brief_summary = desc.get('briefSummary', '')
        detailed_description = desc.get('detailedDescription', '')

        # Status module
        status = protocol.get('statusModule', {})
        overall_status = status.get('overallStatus', '')
        start_date = status.get('startDateStruct', {}).get('date', '')
        completion_date = status.get('completionDateStruct', {}).get('date', '')

        # Design module
        design = protocol.get('designModule', {})
        study_type = design.get('studyType', '')
        phases = design.get('phases', [])
        phase = ', '.join(phases) if phases else ''

        # Conditions module
        conditions = protocol.get('conditionsModule', {})
        condition_list = conditions.get('conditions', [])
        keyword_list = conditions.get('keywords', [])

        # Arms and interventions
        arms = protocol.get('armsInterventionsModule', {})
        interventions = arms.get('interventions', [])

        # Eligibility
        eligibility = protocol.get('eligibilityModule', {})
        eligibility_criteria = eligibility.get('eligibilityCriteria', '')
        min_age = eligibility.get('minimumAge', '')
        max_age = eligibility.get('maximumAge', '')
        sex = eligibility.get('sex', '')

        # Outcomes
        outcomes = protocol.get('outcomesModule', {})
        primary_outcomes = outcomes.get('primaryOutcomes', [])
        secondary_outcomes = outcomes.get('secondaryOutcomes', [])

        # Sponsor
        sponsor = protocol.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsor.get('leadSponsor', {}).get('name', '')

        return {
            'nct_id': nct_id,
            'title': title,
            'org_study_id': org_study_id,
            'status': overall_status,
            'phase': phase,
            'study_type': study_type,
            'conditions': condition_list,
            'keywords': keyword_list,
            'brief_summary': brief_summary,
            'detailed_description': detailed_description,
            'interventions': interventions,
            'eligibility_criteria': eligibility_criteria,
            'min_age': min_age,
            'max_age': max_age,
            'sex': sex,
            'primary_outcomes': primary_outcomes,
            'secondary_outcomes': secondary_outcomes,
            'start_date': start_date,
            'completion_date': completion_date,
            'lead_sponsor': lead_sponsor
        }

    def format_content(self, study: Dict[str, Any]) -> str:
        """
        Format study data as content string.

        Args:
            study: Study data from fetch_study

        Returns:
            Formatted content string
        """
        sections = []

        sections.append(f"Clinical Trial: {study.get('nct_id', 'Unknown')}")
        sections.append("")
        sections.append(f"Official Title: {study.get('title', 'N/A')}")
        sections.append(f"Status: {study.get('status', 'N/A')}")

        if study.get('phase'):
            sections.append(f"Phase: {study['phase']}")

        if study.get('study_type'):
            sections.append(f"Study Type: {study['study_type']}")

        # Conditions
        if study.get('conditions'):
            sections.append(f"Conditions: {', '.join(study['conditions'])}")

        # Brief summary
        if study.get('brief_summary'):
            sections.append("")
            sections.append("Brief Summary:")
            sections.append(study['brief_summary'])

        # Interventions
        if study.get('interventions'):
            sections.append("")
            sections.append("Interventions:")
            for i, interv in enumerate(study['interventions'], 1):
                interv_type = interv.get('type', 'Unknown')
                interv_name = interv.get('name', 'Unknown')
                interv_desc = interv.get('description', '')
                sections.append(f"{i}. {interv_type}: {interv_name}")
                if interv_desc:
                    sections.append(f"   Description: {interv_desc}")

        # Eligibility
        if study.get('eligibility_criteria'):
            sections.append("")
            sections.append("Eligibility Criteria:")
            sections.append(study['eligibility_criteria'][:1000])  # Truncate if very long
            if len(study.get('eligibility_criteria', '')) > 1000:
                sections.append("...")

        # Outcomes
        if study.get('primary_outcomes'):
            sections.append("")
            sections.append("Primary Outcomes:")
            for outcome in study['primary_outcomes'][:3]:  # Limit to 3
                measure = outcome.get('measure', '')
                time_frame = outcome.get('timeFrame', '')
                sections.append(f"- {measure}")
                if time_frame:
                    sections.append(f"  Time Frame: {time_frame}")

        # Dates
        sections.append("")
        if study.get('start_date'):
            sections.append(f"Start Date: {study['start_date']}")
        if study.get('completion_date'):
            sections.append(f"Estimated Completion: {study['completion_date']}")

        # Sponsor
        if study.get('lead_sponsor'):
            sections.append(f"Sponsor: {study['lead_sponsor']}")

        sections.append("")
        sections.append("[Source: ClinicalTrials.gov API v2]")

        return '\n'.join(sections)

    async def fetch_from_url(self, url: str) -> Optional[str]:
        """
        Fetch and format study from URL.

        Args:
            url: ClinicalTrials.gov URL

        Returns:
            Formatted content string
        """
        nct_id = self.extract_nct_id(url)
        if not nct_id:
            logger.warning("No NCT ID found in URL", url=url)
            return None

        study = await self.fetch_study(nct_id)
        if not study:
            return None

        return self.format_content(study)

    async def search_studies(
        self,
        query: str = None,
        condition: str = None,
        intervention: str = None,
        status: str = None,
        phase: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for clinical trials.

        Args:
            query: Free text query
            condition: Condition/disease filter
            intervention: Intervention filter
            status: Trial status filter
            phase: Phase filter
            limit: Maximum results

        Returns:
            List of study summaries
        """
        # Build query parameters
        params = {
            'format': 'json',
            'pageSize': limit
        }

        # Build filter.advanced query
        filters = []
        if query:
            filters.append(query)
        if condition:
            filters.append(f'AREA[Condition]{condition}')
        if intervention:
            filters.append(f'AREA[Intervention]{intervention}')
        if status:
            filters.append(f'AREA[OverallStatus]{status}')
        if phase:
            filters.append(f'AREA[Phase]{phase}')

        if filters:
            params['filter.advanced'] = ' AND '.join(filters)

        url = f"{self.BASE_URL}/studies"

        logger.debug("Searching clinical trials", params=params)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)

                if response.status_code != 200:
                    logger.warning("Clinical trials search failed", status=response.status_code)
                    return []

                data = response.json()
                studies = data.get('studies', [])

                results = []
                for study in studies:
                    protocol = study.get('protocolSection', {})
                    ident = protocol.get('identificationModule', {})
                    status_mod = protocol.get('statusModule', {})
                    design = protocol.get('designModule', {})

                    results.append({
                        'nct_id': ident.get('nctId', ''),
                        'title': ident.get('briefTitle', ''),
                        'status': status_mod.get('overallStatus', ''),
                        'phase': ', '.join(design.get('phases', []))
                    })

                logger.debug("Clinical trials search completed", results=len(results))
                return results

            except Exception as e:
                logger.error("Clinical trials search error", error=str(e))
                return []

    def is_clinical_trials_url(self, url: str) -> bool:
        """Check if URL is from ClinicalTrials.gov."""
        return 'clinicaltrials.gov' in url.lower()
