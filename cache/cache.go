package cache

import "time"

// Setter sets an should set an item in the cache, if the duration is zero or negative the value is expected to be cached forever
var Setter func(key, value string, duration time.Duration) error

// Getter should return ("", nil) if the key is not set
var Getter func(key string) (string, error)

func Get(key string) (string, error) {
	if Getter != nil {
		return Getter(key)
	}
	return "", nil
}

func Set(key, value string, duration time.Duration) error {
	if Setter != nil {
		return Setter(key, value, duration)
	}
	return nil
}
