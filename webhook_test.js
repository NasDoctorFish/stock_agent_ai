whatsapp_json = {
  messaging_product: "whatsapp",
  to: {{phone}},
  type: "template",
  template: {
    name: "jaspers_market_order_confirmation_v1",
    language: { code: "en_US" },
    components: [
      {
        type: "body",
        parameters: [
          { type: "text", text: "Juwon" },
          { type: "text", text: "10 person" },
          { type: "text", text: "Jan 17, 2026" },
        ],
      },
    ],
  },
};

curl -H "Content-Type: application/json" -d '{"name":"John", "age":30}' "https://lee176.app.n8n.cloud/webhook-test/ea51d983-b434-43d6-81f3-0e8b678815e1?hub.mode=subscribe&hub.verify_token=my_whatsapp_verify_123&hub.challenge=123456"




{
  "messaging_product": "whatsapp",
  "to": "6594651111",
  "type": "text",
  "text": {
    "body": "{{ $json.reply }}"
  },

  "meta": {
    "sessionId": "session-1212151241231231241",
    "phone": "6594651111",
    "SYSTEM": "SYSTEM: ...prompt",
    "USER": "USER: ...from prompt",
    "phoneNumberId": "12344515162",
    "customerName": "Juwon",
    "numOfPeople": "10",
    "orderDate": "Jan 10, 2025",
    "orderDetail": "Circular table needed"
  }
}

