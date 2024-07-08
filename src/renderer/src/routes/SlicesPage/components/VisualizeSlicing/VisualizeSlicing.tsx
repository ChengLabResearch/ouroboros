import styles from './VisualizeSlicing.module.css'

import { Canvas, Vector3 } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useMemo } from 'react'
import {
	BoxGeometry,
	BufferAttribute,
	BufferGeometry,
	DoubleSide,
	Float32BufferAttribute
} from 'three'

export type Point = number[]

export type Rect = {
	topLeft: Point
	topRight: Point
	bottomRight: Point
	bottomLeft: Point
}

export type BoundingBox = {
	min: Point
	max: Point
}

const colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

// TODO maybe add grid

// https://github.com/We-Gold/ouroboros/blob/b67033cf3155a5ee6a3356f649b5df84d667fb6c/ouroboros/pipeline/render_slices_pipeline.py
function VisualizeSlicing({
	rects,
	boundingBoxes,
	linkRects,
	useEveryNthRect
}: {
	rects: Rect[]
	boundingBoxes: BoundingBox[]
	linkRects: number[]
	useEveryNthRect?: number
}): JSX.Element {
	if (useEveryNthRect === undefined) {
		useEveryNthRect = 1
	}

	const bounds = useMemo(
		() =>
			boundingBoxes.reduce(
				(acc, { min, max }) => {
					acc.min[0] = Math.min(acc.min[0], min[0])
					acc.min[1] = Math.min(acc.min[1], min[1])
					acc.min[2] = Math.min(acc.min[2], min[2])

					acc.max[0] = Math.max(acc.max[0], max[0])
					acc.max[1] = Math.max(acc.max[1], max[1])
					acc.max[2] = Math.max(acc.max[2], max[2])

					return acc
				},
				{
					min: [Infinity, Infinity, Infinity],
					max: [-Infinity, -Infinity, -Infinity]
				}
			),
		[boundingBoxes]
	)

	const center = [
		(bounds.min[0] + bounds.max[0]) / 2,
		(bounds.min[1] + bounds.max[1]) / 2,
		(bounds.min[2] + bounds.max[2]) / 2
	] as Vector3
	const width = bounds.max[0] - bounds.min[0]
	const height = bounds.max[1] - bounds.min[1]
	const depth = bounds.max[2] - bounds.min[2]

	// Calculate the distance required to view the entire bounding box
	// This is a simplified approach and might need adjustments based on FOV and aspect ratio
	const FOV = 50
	const maxDimension = Math.max(width, height, depth)
	const distance = maxDimension / (2 * Math.tan((Math.PI / 180) * (FOV / 2)))

	// Adjust the camera position to be centered and far enough to see everything
	// Adding some extra distance to ensure the entire bounding box is visible
	const cameraPosition = [
		center[0],
		center[1],
		center[2] + distance + maxDimension * 0.5
	] as Vector3

	return (
		<div className={styles.visualizeSlicing}>
			<Canvas>
				<PerspectiveCamera fov={FOV} makeDefault position={cameraPosition} />
				<OrbitControls target={center as Vector3} />
				{rects.map((rect, i) => {
					if (i % useEveryNthRect !== 0) {
						return null
					}
					const color = colors[linkRects[i] % colors.length]
					return <Slice key={i} rect={rect} color={color} opacity={0.5} />
				})}
				{boundingBoxes.map((boundingBox, i) => {
					const color = colors[i % colors.length]
					return <BoundingBox key={i} boundingBox={boundingBox} color={color} />
				})}
			</Canvas>
		</div>
	)
}

function BoundingBox({
	boundingBox,
	color
}: {
	boundingBox: BoundingBox
	color: string
}): JSX.Element {
	const { min, max } = boundingBox
	const width = max[0] - min[0]
	const height = max[1] - min[1]
	const depth = max[2] - min[2]

	const position = useMemo(
		() => [min[0] + width / 2, min[1] + height / 2, min[2] + depth / 2] as Vector3,
		[boundingBox]
	)

	const geometry = useMemo(() => new BoxGeometry(width, height, depth), [boundingBox])

	return (
		<mesh position={position}>
			<lineSegments>
				<edgesGeometry attach="geometry" args={[geometry]} />
				<lineBasicMaterial attach="material" linewidth={3} />
			</lineSegments>
			<meshBasicMaterial color={color} />
		</mesh>
	)
}

function Slice({
	rect,
	color,
	opacity
}: {
	rect: Rect
	color: string
	opacity: number
}): JSX.Element {
	const geometry = useMemo(() => {
		const vertices = new Float32Array([
			...rect.topRight,
			...rect.topLeft,
			...rect.bottomLeft,
			...rect.bottomRight
		])

		const geom = new BufferGeometry()
		const indices = new Uint16Array([0, 1, 2, 2, 3, 0])

		geom.setAttribute('position', new Float32BufferAttribute(vertices, 3))
		geom.setIndex(new BufferAttribute(indices, 1))
		geom.computeVertexNormals()

		return geom
	}, [rect])

	return (
		<mesh geometry={geometry}>
			<meshBasicMaterial
				transparent={true}
				opacity={opacity}
				color={color}
				side={DoubleSide}
			/>
		</mesh>
	)
}

export default VisualizeSlicing
