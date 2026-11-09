// EventSource stand-in for Node. Test files add the emit shape their stream uses.
export class FakeEventSource extends EventTarget {
	static instances = []
	closeCount = 0
	constructor(url) {
		super()
		this.url = url
		FakeEventSource.instances.push(this)
	}
	close() {
		this.closeCount++
	}
	dispatchMessage(type, payload, lastEventId) {
		this.dispatchEvent(new MessageEvent(type, { data: JSON.stringify(payload), lastEventId }))
	}
}
