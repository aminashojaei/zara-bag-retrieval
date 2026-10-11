from .config import MODEL_LABELS
from .service import VisualSearchService


def build_demo(service: VisualSearchService):
    import gradio as gr

    query_choices = [
        (f"{row.product_id}  |  {row.path.name}", row.relative_key)
        for row in service.store.query_records
    ]
    model_choices = [(MODEL_LABELS[key], key) for key in service.available]
    initial_query = query_choices[0][1]
    initial_model = "siglip2" if "siglip2" in service.available else service.available[0]

    def to_outputs(hits, query_product=None):
        gallery = []
        rows = []
        for rank, hit in enumerate(hits, start=1):
            caption = f"#{rank} · {hit.product_id} · similarity {hit.score:.3f}"
            gallery.append((hit.image_path, caption))
            rows.append([rank, hit.product_id, round(hit.score, 4), hit.image_path])
        extra = ""
        if query_product:
            correct = [i for i, hit in enumerate(hits, start=1) if hit.product_id == query_product]
            extra = f" | Matching product: rank {correct[0]}" if correct else " | Matching product: not in top 3"
        status = (
            f"**{len(hits)} products**{extra}"
        )
        return gallery, rows, status

    def run_saved(key, model):
        if not key or not model:
            raise gr.Error("Select a query and a model.")
        try:
            return to_outputs(service.search_saved_query(model, key), service.query_by_key[key].product_id)
        except Exception as exc:
            raise gr.Error(f"Search failed: {exc}") from exc

    def run_upload(image, model):
        if image is None:
            raise gr.Error("Please upload an image first.")
        if not model:
            raise gr.Error("Select a model first.")
        try:
            return to_outputs(service.search_uploaded_image(model, image))
        except Exception as exc:
            raise gr.Error(f"Search failed: {exc}") from exc

    def run_text(prompt):
        if prompt is None or not str(prompt).strip():
            raise gr.Error("Please enter a text prompt.")
        try:
            return to_outputs(service.search_text_query(prompt))
        except Exception as exc:
            raise gr.Error(f"Search failed: {exc}") from exc

    def preview_query(key):
        return service.get_query_image_path(key) if key else None

    with gr.Blocks(title="Zara Bag Retrieval") as demo:
        gr.Markdown(
            "# Zara Bag Retrieval"
        )
        with gr.Row():
            model = gr.Dropdown(
                choices=model_choices, value=initial_model, interactive=True,
                label="Vision model", 
            )
            gr.Markdown(
                f"**Catalog:** {len({r.product_id for r in service.store.gallery_records})} products  "
                f"\n**Gallery:** {len(service.store.gallery_records)} images  "
                f"\n**Saved queries:** {len(service.store.query_records)}"
            )
        with gr.Tabs():
            with gr.Tab("Choose a saved query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        selected = gr.Dropdown(
                            choices=query_choices, value=initial_query, interactive=True,
                            label="Select a catalog query", filterable=True,
                        )
                        search_saved = gr.Button("Find similar products", variant="primary")
                    with gr.Column(scale=1):
                        preview = gr.Image(
                            value=service.get_query_image_path(initial_query),
                            label="Selected query image", interactive=False, height=250,
                        )
                selected.change(preview_query, inputs=selected, outputs=preview)
            with gr.Tab("Upload your own image"):
                uploaded = gr.Image(type="pil", label="Upload a handbag image", height=290)
                search_upload = gr.Button("Search uploaded image", variant="primary")
            with gr.Tab("Search by text"):
                text_query = gr.Textbox(
                    label="Describe the bag you want to find",
                    placeholder="Example: black leather shoulder bag with chain strap"
                )
                gr.Markdown("Text search uses SigLIP2.")
                search_text = gr.Button("Search by text", variant="primary")
        gr.Markdown("### Top 3 products")
        visual_results = gr.Gallery(
            label="Matching products", columns=3, object_fit="contain",
            height=300, show_label=True,
        )
        status = gr.Markdown("Select a query or upload an image.")
        result_table = gr.Dataframe(
            headers=["Rank", "Product ID", "Cosine similarity", "Gallery image"],
            datatype=["number", "str", "number", "str"],
            label="Rankings", interactive=False,
        )
        search_saved.click(run_saved, inputs=[selected, model],
                           outputs=[visual_results, result_table, status],
                           concurrency_limit=1, concurrency_id="model_inference")
        search_upload.click(run_upload, inputs=[uploaded, model],
                            outputs=[visual_results, result_table, status],
                            concurrency_limit=1, concurrency_id="model_inference")
        search_text.click(run_text, inputs=[text_query],
                          outputs=[visual_results, result_table, status],
                          concurrency_limit=1, concurrency_id="model_inference")
    return demo
