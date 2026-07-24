db = db.getSiblingDB("fraudDetection")

db.paysimData.createIndex(
    { step: 1 },
    { name: "step_index" }
)

db.paysimData.createIndex(
    { isFraud: 1 },
    { name: "fraud_index" }
)

db.paysimData.createIndex(
    { type: 1 },
    { name: "type_index" }
)

db.paysimData.createIndex(
    { amount: -1 },
    { name: "amount_desc_index" }
)

db.paysimData.createIndex(
    {
        isFraud: 1,
        type: 1,
        amount: -1
    },
    {
        name: "fraud_type_amount_index"
    }
)