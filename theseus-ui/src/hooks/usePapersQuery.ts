/**
 * Infinite-scroll papers query (F5).
 *
 * Replaces Papers.tsx's manual fetch state (allPapers / loading /
 * loadingMore / currentPage / hasNextPage / initialLoadComplete) with
 * useInfiniteQuery. The query key carries every server-affecting input,
 * so a filter/sort/profile change resets pagination automatically —
 * the job resetAndFetch used to do.
 */
import { useInfiniteQuery } from '@tanstack/react-query';
import { papersApi } from '../services/api';
import type { PaperApiResponse } from '../services/api';

export interface FilterState {
  minScore: number;
  maxScore: number;
  fromDate: Date | null;
  toDate: Date | null;
  search: string;
  topicId: number | null;
  profileInterestId: number | null;  // For timeline navigation
  // Profile-aware filters (only available when profiles are selected)
  minProfileScore: number;
  maxProfileScore: number;
  relevanceFilter: 'all' | 'relevant' | 'not_relevant';
}

export interface SortCriterion {
  id: string;
  field: 'score' | 'profile_score' | 'date';
  direction: 'asc' | 'desc';
}

export interface PapersQueryParams {
  appliedFilters: FilterState;
  sortCriteria: SortCriterion[];
  useHybridSearch: boolean;
  semanticWeight: number;
  keywordWeight: number;
  selectedProfileIds: number[];
  pageSize: number;
}

interface PapersPage {
  items: PaperApiResponse[];
  current_page: number;
  nextPage?: number | null;
}

const getSortFieldAndDirection = (sortOption: SortCriterion): [string, string] => {
  switch (sortOption.field) {
    case 'score':
      return ['score', sortOption.direction];
    case 'profile_score':
      return ['profile_score', sortOption.direction];
    case 'date':
      return ['date', sortOption.direction];
    default:
      return ['score', 'desc'];
  }
};

async function fetchPapersPage(params: PapersQueryParams, page: number): Promise<PapersPage> {
  const { appliedFilters, sortCriteria, useHybridSearch, semanticWeight,
          keywordWeight, selectedProfileIds, pageSize } = params;
  // Use only the first sort criterion for API call (backend limitation)
  const [sortField, sortDirection] = getSortFieldAndDirection(sortCriteria[0]);

  if (useHybridSearch && appliedFilters.search) {
    // Use hybrid search when enabled and search query exists
    const hybridData = await papersApi.hybridSearch(
      appliedFilters.search,
      page,
      pageSize,
      semanticWeight,
      keywordWeight,
      0.3, // similarity threshold
      appliedFilters.minScore > 0 ? appliedFilters.minScore : undefined,
      appliedFilters.maxScore < 10 ? appliedFilters.maxScore : undefined,
      appliedFilters.fromDate ? appliedFilters.fromDate.toISOString().split('T')[0] : undefined,
      appliedFilters.toDate ? appliedFilters.toDate.toISOString().split('T')[0] : undefined,
      selectedProfileIds.length ? selectedProfileIds : undefined,
      appliedFilters.minProfileScore,
      appliedFilters.maxProfileScore,
      appliedFilters.relevanceFilter === 'all' ? undefined : appliedFilters.relevanceFilter === 'relevant'
    );
    return {
      items: hybridData.results,
      current_page: hybridData.current_page,
      nextPage: hybridData.current_page < hybridData.total_pages ? hybridData.current_page + 1 : null,
    };
  }

  // Use regular search/filtering
  return papersApi.getPapers(
    page,
    pageSize,
    sortField,
    sortDirection,
    appliedFilters.minScore > 0 ? appliedFilters.minScore : undefined,
    appliedFilters.maxScore < 10 ? appliedFilters.maxScore : undefined,
    appliedFilters.fromDate ? appliedFilters.fromDate.toISOString().split('T')[0] : undefined,
    appliedFilters.toDate ? appliedFilters.toDate.toISOString().split('T')[0] : undefined,
    appliedFilters.search || undefined,
    appliedFilters.topicId || undefined,
    selectedProfileIds.length > 0 ? selectedProfileIds : undefined,
    // Profile-aware parameters
    selectedProfileIds.length > 0 && appliedFilters.minProfileScore > 0 ? appliedFilters.minProfileScore : undefined,
    selectedProfileIds.length > 0 && appliedFilters.maxProfileScore < 10 ? appliedFilters.maxProfileScore : undefined,
    // Fixed profile relevance filtering logic
    selectedProfileIds.length > 0 && appliedFilters.relevanceFilter === 'relevant' ? true : undefined,
    // Profile interest ID (from timeline navigation)
    appliedFilters.profileInterestId || undefined
  );
}

export function papersQueryKey(params: PapersQueryParams) {
  return ['papers', {
    ...params.appliedFilters,
    fromDate: params.appliedFilters.fromDate?.toISOString() ?? null,
    toDate: params.appliedFilters.toDate?.toISOString() ?? null,
    sort: params.sortCriteria[0],
    useHybridSearch: params.useHybridSearch,
    semanticWeight: params.semanticWeight,
    keywordWeight: params.keywordWeight,
    profileIds: params.selectedProfileIds,
    pageSize: params.pageSize,
  }] as const;
}

export function usePapersInfiniteQuery(params: PapersQueryParams) {
  return useInfiniteQuery({
    queryKey: papersQueryKey(params),
    queryFn: ({ pageParam }) => fetchPapersPage(params, pageParam),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => lastPage.nextPage ?? null,
  });
}
