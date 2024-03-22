<script lang="ts">
    import { page } from "$app/stores";

    import ResultCard from "$lib/components/search_item.svelte";
    import Pagenation from "$lib/components/pagenate.svelte";
    import SearchBox from "$lib/components/searchbox.svelte";
    import type { SearchData } from "$lib/type/types";
    import { getPagenationParams, getSearchOptParams } from "$lib/utils/util";

    export let data: SearchData;
    let query = data.query;
    $: search_result = data.posts;
    $: pagenation_params = getPagenationParams($page.url.searchParams);

    let search_opt_params = getSearchOptParams($page.url.searchParams);
    let topk = search_opt_params.topk;
    const term_filter = search_opt_params.hybrid;
    let search_fields = search_opt_params.search_fields;

    function buildSearchAPIParams(): string {
        const search_params = {
            query: data.query,
            topk: `${topk}`,
            hybrid: `${Number(term_filter)}`,
            search_fields: search_fields.join(","),
        };

        const searchParams = new URLSearchParams(search_params);
        return searchParams.toString();
    }
</script>

<div class="container mx-auto flex justify-center items-center">
    <div class="space-y-4 text-center flex flex-col items-center w-10/12">
        <SearchBox {topk} {term_filter} {query} {search_fields} />
        {#if search_result.length > 0}
            {#each search_result as item}
                <ResultCard {item} />
            {/each}
            <Pagenation
                params={pagenation_params}
                rest_param={buildSearchAPIParams()}
                total={data.total}
            />
        {:else}
            <div class="text-center space-y-4">
                <h2 class="h2">No results found</h2>
                <p class="text-gray-500">
                    We couldn't find any results matching{" "}
                    <span class="font-medium">{data.query}</span>
                </p>
            </div>
        {/if}
    </div>
</div>

<style lang="postcss">
</style>
