import { JSX } from "react"

function PluginDisplay({ url }: { url: string }): JSX.Element {
	return (
		<iframe
			src={url}
			style={{ width: '100%', height: '100%', border: 'none' }}
			scrolling="no"
			allowFullScreen
		/>
	)
}

export default PluginDisplay
