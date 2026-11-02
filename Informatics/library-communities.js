"use strict";
(() => {
  const labels = {direct:"Direct population / topic",context:"Context discussed",related:"Related research · inferred",uncertain:"Not established from available text","no-match":"No supported match in checked text"};
  function attach(record, data) {
    const ids = new Set([record.id, ...(record.aliases || [])]);
    record.communityAssessments = data.records.filter(item => ids.has(item.recordId));
    record.communityMatches = record.communityAssessments.flatMap(item => item.matches);
    if (record.communityAssessments.some(item => item.reviewStatus)) {
      record.communityReview = record.communityAssessments.every(item => item.reviewStatus === "approved") ? "reviewed" : "pending";
    } else delete record.communityReview;
    return record;
  }
  function matches(record, category, includeRelated = false) {
    return (record.communityMatches || []).some(item => item.categoryId === category && (item.relation === "direct" || item.relation === "context" || (includeRelated && item.relation === "related")));
  }
  function terms(record, taxonomy) {
    const ids = new Set((record.communityMatches || []).filter(item => ["direct","context"].includes(item.relation)).map(item => item.categoryId));
    return taxonomy.filter(item => ids.has(item.id)).flatMap(item => [item.label, ...item.searchTerms]).join(" ");
  }
  globalThis.UniversalResearchCommunities = {attach,matches,terms,labels};
})();
