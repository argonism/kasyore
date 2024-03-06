<script lang="ts">
    import { goto } from "$app/navigation";
    import SearchBox from "$lib/components/searchbox.svelte";

    let topk: number = 100;
    let term_filter: boolean = true;
    let search_box = "";
    const onSearchKeyDown = function (event: KeyboardEvent) {
        if (event.key === "Enter") {
            const search_params = {
                query: search_box,
                topk: `${topk}`,
            };

            const searchParams = new URLSearchParams(search_params);
            goto("/search?" + searchParams.toString());
        }
    };
</script>

<div class="container h-full mx-auto flex justify-center items-center">
    <div class="space-y-4 text-center flex flex-col items-center w-10/12">
        <SearchBox {topk} {term_filter} />
    </div>
</div>

<style lang="postcss">
</style>
