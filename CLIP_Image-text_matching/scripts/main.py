import os
import gradio as gr
from scripts.insert_search import image_search  # 确保导入路径正确


def main():
    gr.close_all()  # 关闭之前的所有实例
    # 创建Blocks界面
    app = gr.Blocks(
        theme=gr.themes.Default(),
        title="Image Search System",
        css=".gradio-container {background-color: #FFD1DC;} "
            ".gradio-container button {background-color: #FFD1DC;} "
            "footer {visibility: hidden}"
    )

    with app:
        with gr.Tabs():
            with gr.TabItem("Image Search"):
                with gr.Row():
                    with gr.Column():
                        # 文本输入框
                        text = gr.TextArea(
                            label="Text Description",
                            placeholder="Enter image description here...",
                            value=""
                        )
                        # 图像输入组件
                        img_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=200
                        )
                        # 搜索按钮
                        btn = gr.Button(
                            value="Search",
                            variant="primary"
                        )

                    with gr.Column():
                        with gr.Row():
                            # 输出图像组件
                            output_images = [
                                gr.Image(type="pil", label=None, height=150, width=150)
                                for _ in range(6)
                            ]

            # 绑定按钮点击事件
            btn.click(
                fn=image_search,
                inputs=[text, img_input],
                outputs=output_images,
                show_progress=True
            )

    # 启动界面 - 移除 concurrency_limit 参数，使用默认队列配置
    ip_addr = '0.0.0.0'
    app.queue().launch(  # 仅保留 queue()，不传递参数
        show_api=False,
        share=False,
        server_name=ip_addr,
        server_port=6006
    )


if __name__ == '__main__':
    main()
    #http://127.0.0.1:6006